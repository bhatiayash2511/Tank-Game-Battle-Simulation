#include <iostream>
using namespace std;

struct Tank {
    int x;
    int y;
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
    Tank linePoint1 = {2, 1};
    Tank linePoint2 = {200, 100};

    Tank testPoints[10] = {
        {5, 2}, {10, 5}, {50, 20}, {100, 40}, {150, 60},
        {200, 80}, {250, 100}, {300, 120}, {350, 140}, {400, 160}
    };

    for (const Tank& testPoint : testPoints) {
        bool result = isPointOnLine(testPoint, linePoint1, linePoint2);
        cout << "Tank (" << testPoint.x << ", " << testPoint.y << ") is on the line? " << boolalpha << result << endl;
    }

    return 0;
}
