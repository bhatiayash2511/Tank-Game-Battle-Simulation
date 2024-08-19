#include <iostream>
#include <vector>
#include <algorithm> // for min and max
using namespace std;

struct Tank {
    int x, y; // Tank position
    int hp;   // Health Points
    int score; // Score
};

bool isPointOnLine(const Tank& point, const Tank& linePoint1, const Tank& linePoint2) {
    int x = point.x, y = point.y;
    int x1 = linePoint1.x, y1 = linePoint1.y;
    int x2 = linePoint2.x, y2 = linePoint2.y;

    // Check if the line is vertical
    if (x1 == x2) {
        return x == x1;  // Tank lies on the line if x-coordinate matches
    }
    else if (y1 == y2) {
        return y == y1;  // Tank lies on the line if y-coordinate matches
    }
    else {
        // Calculate slopes using type casting to ensure floating-point division
        double slope1 = static_cast<double>(y - y1) / (x - x1);
        double slope2 = static_cast<double>(y2 - y1) / (x2 - x1);

        // Compare slopes to check if they are equal
        return slope1 == slope2;
    }
}

// Function to simulate the battlefield until one or zero tanks have health
void simulateBattlefield(int M, int N, int T, vector<Tank>& tanks) {
    int tanksLeft = T;
    int round = 1;
    while (tanksLeft > 1) {  // Continue rounds until only one or zero tanks left
        if(round % T == 0){
            round++;
            continue;
        }
        vector <int> tankHitC(T, 0);
        for (int tank_id = 0; tank_id < T; ++tank_id) {
            if (tanks[tank_id].hp <= 0) {
                continue; // Skip destroyed tanks
            }

            int target_id = (tank_id + round) % T;  // Choose the next tank as the target

            // if (tanks[target_id].hp <= 0) {
            //     continue; // Skip targets with zero health
            // }
            
            Tank& linePoint1 = tanks[tank_id];
            Tank& linePoint2 = tanks[target_id];
            int dir = 0;
            if(linePoint1.x <= linePoint2.x && linePoint1.y <= linePoint2.y){
                dir = 1;
            }
            if(linePoint1.x > linePoint2.x && linePoint1.y <= linePoint2.y){
                dir = 2;
            }
            if(linePoint1.x > linePoint2.x && linePoint1.y > linePoint2.y){
                dir = 3;
            }
            if(linePoint1.x <= linePoint2.x && linePoint1.y > linePoint2.y){
                dir = 4;
            }
            int it = -1;
            int temp = 1e9;
            for (int i = 0; i < T; i++) {
                if(i == tank_id){ 
                    continue;
                }
                Tank& testPoint = tanks[i];
                if(dir == 1){
                    if (testPoint.hp > 0 && tanks[tank_id].x <=  testPoint.x && tanks[tank_id].y <=  testPoint.y) {
                        bool result = isPointOnLine(testPoint, linePoint1, linePoint2);
                        // cout << "Tank (" << testPoint.x << ", " << testPoint.y << ") is on the line? " << boolalpha << result << endl;
                        if (result && (abs(linePoint1.x - testPoint.x) + abs(linePoint1.y - testPoint.y)) < temp) {
                            temp = (abs(linePoint1.x - testPoint.x) + abs(linePoint1.y - testPoint.y));
                            it = i;
                        }
                    }
                }
                if(dir == 2){
                    if (testPoint.hp > 0 && tanks[tank_id].x > testPoint.x && tanks[tank_id].y <= testPoint.y) {
                        bool result = isPointOnLine(testPoint, linePoint1, linePoint2);
                        // cout << "Tank (" << testPoint.x << ", " << testPoint.y << ") is on the line? " << boolalpha << result << endl;
                        if (result && (abs(linePoint1.x - testPoint.x) + abs(linePoint1.y - testPoint.y)) < temp) {
                            temp = (abs(linePoint1.x - testPoint.x) + abs(linePoint1.y - testPoint.y));
                            it = i; 
                        }
                    }
                }
                if(dir == 3){
                    if (testPoint.hp > 0 && tanks[tank_id].x > testPoint.x && tanks[tank_id].y > testPoint.y) {
                        bool result = isPointOnLine(testPoint, linePoint1, linePoint2);
                        // cout << "Tank (" << testPoint.x << ", " << testPoint.y << ") is on the line? " << boolalpha << result << endl;
                        if (result && (abs(linePoint1.x - testPoint.x) + abs(linePoint1.y - testPoint.y)) < temp) {
                            temp = (abs(linePoint1.x - testPoint.x) + abs(linePoint1.y - testPoint.y));
                            it = i;
                        }
                    }
                }
                if(dir == 4){
                    if (testPoint.hp > 0 && tanks[tank_id].x <= testPoint.x && tanks[tank_id].y > testPoint.y) {
                        bool result = isPointOnLine(testPoint, linePoint1, linePoint2);
                        // cout << "Tank (" << testPoint.x << ", " << testPoint.y << ") is on the line? " << boolalpha << result << endl;
                        if (result && (abs(linePoint1.x - testPoint.x) + abs(linePoint1.y - testPoint.y)) < temp) {
                            temp = (abs(linePoint1.x - testPoint.x) + abs(linePoint1.y - testPoint.y));
                            it = i;
                        }
                    }
                }

            }
            if(it != -1){ 
                tankHitC[it]++;
                // tanks[it].hp--;
                tanks[tank_id].score++;
                // if(tanks[it].hp == 0) tanksLeft--;
            }
            
        }
        
        for (int i = 0; i < T; ++i) {
            tanks[i].hp -= tankHitC[i];
            if(tanks[i].hp <= 0 && tankHitC[i]) tanksLeft--;
            cout << "Tank " << i << " Score: " << tanks[i].score << " Health: " << tanks[i].hp << endl;
        }
        round++;
    }
}

int main() {
    int M = 5; // Grid size M
    int N = 5; // Grid size N
    int T = 3; // Number of tanks

    vector<Tank> tanks = { {1, 1, 5, 0}, {2, 2, 5, 0}, {3, 3, 5, 0} };

    // Simulate battlefield
    simulateBattlefield(M, N, T, tanks);

    // Print scores

    cout << endl << endl;
    for (int i = 0; i < T; ++i) {
        cout << "Tank " << i << " Score: " << tanks[i].score << " Health: " << tanks[i].hp << endl;
    }

    return 0;
}

// ans = 6 3 6
