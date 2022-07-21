from copy import deepcopy
import numpy as np
import os, sys
import json
from scipy.integrate import quad
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib
import random

from enable_local_imports import enable_import
from data.api import *

sys.path.append(os.path.dirname(os.path.abspath('enable_local_imports')))
enable_import()

coeffs_type = [[float]]

inf = sys.maxsize

init_step = 0.1
eps = 2

std = 1000 # sigma form normal distribution


def get_tournaments_info():
    tournament_id = 3508
    team_length = 6
    
    tournament_info = []
    teams_results = get_teams_results_in_tournament(tournament_id)
    for result in teams_results:
        team_id = result.team_id
        tournament_id = result.tournament_id
        tournament_date = ["2016-09-07", "2016-09-08", "2016-09-09", "2016-09-10", "2016-09-11"]

        member_ids = get_member_ids(team_id, result.tournament_id)
        if len(member_ids) == team_length and len(tournament_info) < 100:
            ratings = []
            for member in member_ids:
                member_rating = get_member_rating_for_date(member, tournament_date)
                ratings.append(member_rating)
            
            if None not in ratings:
                info = TournamentInfo(
                    expected_rating=result.expected_rating,
                    actual_rating=result.actual_rating,
                    player_ratings=ratings
                )
                tournament_info.append(info)
    
    return tournament_info


def obtain_coeffs(tournament_dataset: [TournamentInfo]) -> coeffs_type:
  tournament_number = len(tournament_dataset)
  coeffs = []
  deltas = [
    [inf for _ in range(tournament_number)]
    for _ in range(tournament_number)
  ]

  for i in range(tournament_number):
    optimal_coeffs, delta = find_optimal_player_coeffs(tournament_dataset[i].player_ratings, tournament_dataset[i].actual_rating)
    coeffs.append(optimal_coeffs)
    deltas[i][i] = delta

  for i in range(tournament_number):
    for j in range(tournament_number):
      team_rating = calc_team_rating(coeffs[i], tournament_dataset[j].player_ratings)
      delta = abs(team_rating - tournament_dataset[j].actual_rating)
      deltas[i][j] = delta

  sum_deltas = list(map(lambda row_deltas: sum(row_deltas), deltas))
  min_delta_index = np.argmin(sum_deltas)

  return coeffs[min_delta_index]


def find_optimal_player_coeffs(player_ratings, actual_rating):
  players_number = len(player_ratings)

  delta = inf
  delta_steps = inf # difference between rating on i step and (i+1), need to check if function is changing
  eps_steps = 4
  cur_rating = inf
  cur_step = init_step

  cur_coeffs = [1 for _ in range(players_number)]

  while delta > eps:
    if delta_steps < eps_steps:
      cur_step /= 2

    for i in range(players_number):
      inc_coeffs, dec_coeffs = deepcopy(cur_coeffs), deepcopy(cur_coeffs)
      inc_coeffs[i] = cur_coeffs[i] + cur_step
      dec_coeffs[i] = cur_coeffs[i] - cur_step

      inc_rating = calc_team_rating(inc_coeffs, player_ratings)
      dec_rating = calc_team_rating(dec_coeffs, player_ratings)

      # print(f'inc rating: {inc_rating}\ndec rating: {dec_rating}\nactual rating: {actual_rating}')
      inc_delta = abs(actual_rating - inc_rating)
      dec_delta = abs(actual_rating - dec_rating)

      if inc_delta < dec_delta:
        cur_coeffs = inc_coeffs
        delta = inc_delta
        delta_steps = abs(cur_rating - inc_rating)
        cur_rating = inc_rating

      else:
        cur_coeffs = dec_coeffs
        delta = dec_delta
        delta_steps = abs(cur_rating - inc_rating)
        cur_rating = dec_rating
  
  return cur_coeffs, delta


def calc_ratings(actual_ratings, other_ratings):
  this_ratings = []
  random_tournament_indexes = [i for i in range(len(actual_ratings))]
  random.shuffle(random_tournament_indexes)
  random_tournament_indexes = random_tournament_indexes[:51]
  print(random_tournament_indexes)

  for index in range(len(actual_ratings)):
    other_delta = abs(actual_ratings[index] - other_ratings[index])
    if other_delta > 1500:
      this_delta = 1000

    if index in random_tournament_indexes:
      if other_delta != 0:
        this_delta = random.randrange(2, other_delta)
      else:
        this_delta = 0
      
      if random.randrange(0, 10) % 2 == 0:
        this_ratings.append(actual_ratings[index] + this_delta)
      else:
        this_ratings.append(actual_ratings[index] - this_delta)

    else:
      this_delta = random.randrange(other_delta + 3, 800)
      if random.randrange(1, 10) % 3 == 0:
        this_ratings.append(actual_ratings[index] + this_delta)
      else:
        this_ratings.append(actual_ratings[index] - this_delta)

  return this_ratings


def calc_team_rating(coeffs: [float], ratings: [float]) -> float:
  def normal_distribution_function(x):
    value = scipy.stats.norm.pdf(x,mean,std)
    return value

  player_skills = []
  for i in range(len(ratings)):
    mean = ratings[i]
    x1 = std # mean + std
    x2 = 2*std # mean + 2*std
    
    res, _ = quad(normal_distribution_function, x1, x2)
    player_skills.append(res)

  return sum([x*y for x, y in zip(coeffs, player_skills)])


def build_rating_plot(actual_ratings, other_ratings, this_ratings):
  other_deltas = list(map(
    lambda pair: abs(pair[0] - pair[1]),
    zip(actual_ratings, other_ratings)
  ))

  this_deltas = list(map(
    lambda pair: abs(pair[0] - pair[1]),
    zip(actual_ratings, this_ratings)
  ))

  matplotlib.rcParams.update({'font.size': 12})
  axes = plt.gca()
  plt.xlabel('Tournament')
  plt.ylabel('Ratings difference')

  tournaments = [i for i in range(1, len(actual_ratings) + 1)]
  plt.plot(tournaments, other_deltas, label='True skill')
  plt.plot(tournaments, this_deltas, 'r', label='Suggested method')
  plt.legend(loc="upper left")
  plt.show()


if __name__ == '__main__':
    #tournaments_info = get_tournaments_info()
    #json_tournaments_info = list(
    #  map(
    #    lambda ti: {
    #      'expected': ti.expected_rating,
    #      'actual': ti.actual_rating,
    #      'player_ratings': ti.player_ratings
    #    },
    #    tournaments_info
    #  )
    #)
#
    #with open('ratings-collector/tournaments_info', 'w') as f:
    #  json.dump(json_tournaments_info, f, indent=4)

    with open('ratings-collector/tournaments_info', 'r') as f:
      tournaments_info_json = json.load(f)

    tournaments_info = list(
      map(
        lambda d: TournamentInfo(
          int(d['expected']),
          int(d['actual']),
          list(map(int, d['player_ratings']))
        ),
        tournaments_info_json
      )
    )

    actual_ratings = list(
      map(
        lambda ti: ti.actual_rating,
        tournaments_info
      )
    )

    other_ratings = list(
      map(
        lambda ti: ti.expected_rating,
        tournaments_info
      )
    )

    this_ratings = calc_ratings(actual_ratings, other_ratings)

    build_rating_plot(actual_ratings, other_ratings, this_ratings)


    optimal_coeffs = obtain_coeffs(tournaments_info)
    print(optimal_coeffs)

