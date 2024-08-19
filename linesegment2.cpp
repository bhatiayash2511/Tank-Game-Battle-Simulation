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
        
        cout << "Slope 1: " << slope1 << endl;
        cout << "Slope 2: " << slope2 << endl;

        // Compare slopes to check if they are equal
        return slope1 == slope2;
    }
}

int main() {
    

    Tank pointToCheck = {5, 2};
    Tank linePoint1 = {1, 1};
    Tank linePoint2 = {9, 3};

    bool result = isPointOnLine(pointToCheck, linePoint1, linePoint2);
    cout << boolalpha << "Is point on line? " << result << endl;

    return 0;
}
