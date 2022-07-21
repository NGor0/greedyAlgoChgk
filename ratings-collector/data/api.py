from chgk_rating_client import Rating

from data.types import *


def get_teams_results_in_tournament(tournament_id: int) -> [TeamRating]:
    client = Rating()
    team_results = client.tournament_results(tournament_id)
    
    team_result_parsed = []
    for team_result in team_results:
        team_id = team_result['idteam']
        tech_rating = team_result['tech_rating_rb']
        real_rating = team_result['rating_r']
        
        team_result_parsed.append(
            TeamRating(
                team_id=team_id,
                tournament_id=tournament_id,
                expected_rating=tech_rating,
                actual_rating=real_rating
            )
        )
    client.clear_cache()
    return team_result_parsed


def get_member_ids(team_id, tournament_id) -> [int]:
    client = Rating()
    members_info = client.tournament_roster(tournament_id, team_id)
    
    members_info_ids = []
    for member_info in members_info:
        member_id = member_info['idplayer']
        members_info_ids.append(member_id)
    client.clear_cache()
    return members_info_ids


def get_member_rating_for_date(player_id, t_date) -> int:
    client = Rating()
    members_rating_info = client.player_ratings(player_id)
    for member_rating_info in members_rating_info:
        date = member_rating_info['date']
        rating = member_rating_info['rating']
        if date in t_date:
            return rating
        
    client.clear_cache()


