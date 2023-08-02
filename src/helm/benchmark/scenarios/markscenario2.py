import random
from typing import List

from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output

class MarkScenario2(Scenario):
    """
    A task where the input is a play in the game of Rock, Paper, Scissors and the output
    is the move that would beat the input play.

    Example:

        rock -> paper
        paper -> scissors
        scissors -> rock
    """

    name = "markscenario2"
    description = "Rock, Paper, Scissors game"
    tags = ["markscenario2"]

    def __init__(self, num_train_instances: int, num_test_instances: int, **kwargs):
        super().__init__()
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        self.moves = ['rock', 'paper', 'scissors']
        self.winning_moves = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}

    def get_instances(self) -> List[Instance]:
        random.seed(1)

        def generate_instance(split: str) -> Instance:
            """Generate a random instance with `tags`."""
            input_move: str = random.choice(self.moves)
            output_move: str = self.winning_moves[input_move]
            wrong_move: str = next(move for move in self.moves if move != output_move)

            references: List[Reference] = [
                Reference(Output(text=output_move), tags=[CORRECT_TAG]),  # Correct output
                Reference(Output(text=wrong_move), tags=[]),  # Wrong output
            ]
            return Instance(Input(text=input_move), references=references, split=split)

        def generate_instances(num_instances: int, split: str):
            return [generate_instance(split) for _ in range(num_instances)]

        return generate_instances(self.num_train_instances, TRAIN_SPLIT) + generate_instances(
            self.num_test_instances, TEST_SPLIT
        )
