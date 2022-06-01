#include "gurobi_c++.h"
#include <eigen3/Eigen/Dense>
#include "optimize_learning/SIMPLEX.hpp"
#include "optimize_learning/DUAL_SIMPLEX.hpp"

using namespace std;

void pri(Eigen::MatrixXd m)
{
    std::cout << m << std::endl;
}

int main()
{
    Eigen::MatrixXd A(2, 4);
    A << 2, 1, 1, 0, 1, 2, 0, 1;
    /*A << 1, 0, 2.3, 1,
         0, 1, 1, 2;*/
    Eigen::MatrixXd b(2, 1);
    b << 6, 6;
    //b << 12, 9;
    Eigen::MatrixXd c(1, 4);
    c << -3, -2, 0, 0;
    //c << 0, 0, -1, -10;
    SIMPLEX test(A, b, c);
    DUAL_SIMPLEX test2(A, b, c);
    pri(test.solve());
    pri(test2.solve());
}
