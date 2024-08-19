#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <vector>
using namespace std;

struct Tanks {
    int *x;    // X coordinates of tanks
    int *y;    // Y coordinates of tanks
    int *hp;   // Health Points of tanks
    int *score; // Scores of tanks
};

bool isPointOnLine(const Tanks& tanks, int pointIndex, int linePoint1Index, int linePoint2Index) {
    int x = tanks.x[pointIndex], y = tanks.y[pointIndex];
    int x1 = tanks.x[linePoint1Index], y1 = tanks.y[linePoint1Index];
    int x2 = tanks.x[linePoint2Index], y2 = tanks.y[linePoint2Index];

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

int main(int argc, char **argv) {
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL) {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int));  // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int));  // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++) {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    Tanks tanks;
    tanks.x = (int *)malloc(T * sizeof(int));    // X coordinate of each tank
    tanks.y = (int *)malloc(T * sizeof(int));    // Y coordinate of each tank
    tanks.hp = (int *)malloc(T * sizeof(int));   // Health Points of each tank
    tanks.score = (int *)malloc(T * sizeof(int)); // Score of each tank

    for (int i = 0; i < T; ++i) {
        tanks.x[i] = xcoord[i];
        tanks.y[i] = ycoord[i];
        tanks.score[i] = score[i];
        tanks.hp[i] = H;
    }
    int tanksLeft = T;
    int currentRound = 1;
    while (tanksLeft > 1) {  // Continue rounds until only one or zero tanks left
        if (currentRound % T == 0) {
            currentRound++;
            continue;
        }
        vector<int> tankHitCount(T, 0);
        for (int tank_id = 0; tank_id < T; ++tank_id) {
            if (tanks.hp[tank_id] <= 0) {
                continue; // Skip destroyed tanks
            }

            int target_id = (tank_id + currentRound) % T;  // Choose the next tank as the target

            int x1 = tanks.x[tank_id];
            int y1 = tanks.y[tank_id];
            int x2 = tanks.x[target_id];
            int y2 = tanks.y[target_id];

            int dir = 0;
            if (x1 <= x2 && y1 <= y2) {
                dir = 1;
            }
            if (x1 > x2 && y1 <= y2) {
                dir = 2;
            }
            if (x1 > x2 && y1 > y2) {
                dir = 3;
            }
            if (x1 <= x2 && y1 > y2) {
                dir = 4;
            }
            int it = -1;
            int temp = 1e9;
            for (int i = 0; i < T; i++) {
                if (i == tank_id) {
                    continue;
                }
                if (dir == 1) {
                    if (tanks.hp[i] > 0 && x1 <= tanks.x[i] && y1 <= tanks.y[i]) {
                        bool result = isPointOnLine(tanks, i, tank_id, target_id);
                        if (result && (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i])) < temp) {
                            temp = (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i]));
                            it = i;
                        }
                    }
                }
                if (dir == 2) {
                    if (tanks.hp[i] > 0 && x1 > tanks.x[i] && y1 <= tanks.y[i]) {
                        bool result = isPointOnLine(tanks, i, tank_id, target_id);
                        if (result && (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i])) < temp) {
                            temp = (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i]));
                            it = i;
                        }
                    }
                }
                if (dir == 3) {
                    if (tanks.hp[i] > 0 && x1 > tanks.x[i] && y1 > tanks.y[i]) {
                        bool result = isPointOnLine(tanks, i, tank_id, target_id);
                        if (result && (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i])) < temp) {
                            temp = (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i]));
                            it = i;
                        }
                    }
                }
                if (dir == 4) {
                    if (tanks.hp[i] > 0 && x1 <= tanks.x[i] && y1 > tanks.y[i]) {
                        bool result = isPointOnLine(tanks, i, tank_id, target_id);
                        if (result && (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i])) < temp) {
                            temp = (abs(x1 - tanks.x[i]) + abs(y1 - tanks.y[i]));
                            it = i;
                        }
                    }
                }

            }
            if (it != -1) {
                tankHitCount[it]++;
                tanks.score[tank_id]++;
            }
        }

        for (int i = 0; i < T; ++i) {
            tanks.hp[i] -= tankHitCount[i];
            if (tanks.hp[i] <= 0 && tankHitCount[i]) tanksLeft--;
        }
        currentRound++;
    }

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++) {
        fprintf(outputfilepointer, "%d\n", tanks.score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    free(tanks.x);
    free(tanks.y);
    free(tanks.hp);
    free(tanks.score);
    cudaDeviceSynchronize();
    return 0;
}
