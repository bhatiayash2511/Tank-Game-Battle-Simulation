#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

struct Tank {
    int id;
    int hp;
    int score;
};

vector<int> simulateBattlefield(int M, int N, int T, vector<pair<int, int>> tankCoordinates, vector<int> tankHP) {
    // Initialize grid with tanks
    vector<vector<Tank>> grid(M, vector<Tank>(N, {-1, 0, 0}));
    for (int i = 0; i < T; ++i) {
        int x = tankCoordinates[i].first;
        int y = tankCoordinates[i].second;
        grid[x][y] = {i, tankHP[i], 0};
    }

    // Define directions for firing (up, down, left, right)
    vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    // Function to check if coordinates are within bounds
    auto inBounds = [&](int x, int y) {
        return x >= 0 && x < M && y >= 0 && y < N;
    };

    // Function to simulate firing and update scores
    auto fire = [&](int tankId, int x, int y, pair<int, int> direction) {
        int dx = direction.first;
        int dy = direction.second;
        while (inBounds(x, y)) {
            if (grid[x][y].id != tankId && grid[x][y].id != -1) {
                int targetId = grid[x][y].id;
                grid[targetId].hp -= 1;
                grid[tankId].score += 1;
                if (grid[targetId].hp <= 0) {
                    grid[targetId].id = -1;  // Tank destroyed
                }
                break;
            }
            x += dx;
            y += dy;
        }
    };

    // Simulate rounds
    for (int k = 0; k < T; ++k) {
        for (int i = 0; i < T; ++i) {
            if (grid[tankCoordinates[i].first][tankCoordinates[i].second].id != -1) {  // Tank not destroyed
                int x = tankCoordinates[i].first;
                int y = tankCoordinates[i].second;
                pair<int, int> direction = directions[(i + k) % 4];
                fire(i, x + direction.first, y + direction.second, direction);
            }
        }
    }

    // Get scores at the end of the game
    vector<int> scores;
    for (const auto& row : grid) {
        for (const auto& tank : row) {
            if (tank.id != -1) {
                scores.push_back(tank.score);
            }
        }
    }
    return scores;
}

int main() {
    int M = 5;
    int N = 5;
    int T = 3;
    vector<pair<int, int>> tankCoordinates = {{0, 0}, {2, 2}, {4, 4}};
    vector<int> tankHP = {5, 7, 3};

    vector<int> scores = simulateBattlefield(M, N, T, tankCoordinates, tankHP);

    // Output scores
    cout << "Scores at the end of the game:\n";
    for (int i = 0; i < scores.size(); ++i) {
        cout << "Tank " << i << ": " << scores[i] << endl;
    }

    return 0;
}
