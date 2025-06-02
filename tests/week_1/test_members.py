import os
import unittest


class MyTestCase(unittest.TestCase):
    def test_file_exists(self):
        assert os.path.isfile("rl_exercises/members.txt"), (
            "File members.txt does not exist!"
        )

    def test_members(self):
        with open("rl_exercises/members.txt") as fh:
            lines = fh.readlines()

        (
            self.assertTrue(lines[0].startswith("member 1: ")),
            "First line should start with 'member 1: '",
        )
        (
            self.assertTrue(lines[1].startswith("member 2: ")),
            "Second line should start with 'member 2: '",
        )
        (
            self.assertTrue(lines[2].startswith("member 3: ")),
            "Third line should start with 'member 3: '",
        )

    def test_names_replaced(self):
        with open("rl_exercises/members.txt") as fh:
            lines = fh.readlines()

        self.assertFalse("name1" in lines[0]), "The first name is still a placeholder!"
        self.assertFalse("name2" in lines[1]), "The second name is still a placeholder!"
        self.assertFalse("name3" in lines[2]), "The third name is still a placeholder!"


if __name__ == "__main__":
    unittest.main()
