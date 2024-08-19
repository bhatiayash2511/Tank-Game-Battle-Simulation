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

int main() {
    Tank linePoint1 = {200, 100};
    Tank linePoint2 = {100, 50};

    vector<Tank> tanks = {
        {5, 2}, {10, 5}, {80, 40}, {120, 60}, {150, 75},
        {200, 80}, {250, 100}, {300, 120}, {350, 140}, {300, 150}
    };
    Tank p;
    int lowx = min(linePoint1.x, linePoint2.x);
    int highx = max(linePoint1.x, linePoint2.x);
    int lowy = min(linePoint1.y, linePoint2.y);
    int highy = max(linePoint1.y, linePoint2.y);
    int temp = 1e9;
    for (const Tank& testPoint : tanks) {
        if(testPoint.x <= highx && testPoint.y <= highy && testPoint.x >= lowx && testPoint.y >= lowy){
            bool result = isPointOnLine(testPoint, linePoint1, linePoint2);
            cout << "Tank (" << testPoint.x << ", " << testPoint.y << ") is on the line? " << boolalpha << result << endl;
            if(result && abs(linePoint1.x - testPoint.x) < temp){
                p = testPoint;
                temp = abs(linePoint1.x - testPoint.x);
            }
        }
    }
    cout << "closest Tank (" << p.x << ", " << p.y << ")" << endl;

    return 0;
}
