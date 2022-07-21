from dataclasses import dataclass


@dataclass
class TournamentInfo:
    expected_rating: int
    actual_rating: int
    player_ratings: [int]

@dataclass
class TeamRating:
    team_id: int
    tournament_id: int
    expected_rating: int
    actual_rating: int