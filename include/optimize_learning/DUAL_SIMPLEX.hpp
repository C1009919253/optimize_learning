#pragma once
#include <eigen3/Eigen/Dense>
#include "SIMPLEX.hpp"

using namespace Eigen;

class DUAL_SIMPLEX
{
public:
    DUAL_SIMPLEX(MatrixXd A1, MatrixXd b1, MatrixXd c1)
    {
        A = A1;
        b = b1;
        c = c1;
    }
    MatrixXd solve()
    {
        int m = A.rows();
        int n = A.cols();
        MatrixXd B(m+1, n+1);
        B.setZero(m+1, n+1);
        B.block(1, 0, m, n) = A;
        B.block(0, 0, 1, n) = c;
        B.block(1, n, m, 1) = b;
        MatrixXd B_index(1, m);
        for (int i = 0; i < m; i++)
            B_index(i) = n-m+i+1;
        for (int times = 0; times < 10; times++)
        {
            MatrixXd rhl(m, 1);
            rhl = B.block(1, n, m, 1);
            int rp, cp;
            double s = rhl.minCoeff(&rp, &cp);
            if (s > 0) // it's best now
            {
                MatrixXd X(1, n);
                X.setZero(1, n);
                for (int i = 0; i < m; i++)
                    X(B_index(i)-1) = B(i+1, n);
                return X;
            }
            double min1 = 1.0/0.0;
            int min1_index = -1;
            for (int i = 0; i < n; i++)
            {
                if (B(rp+1, i) >= 0)
                    continue;
                double xx = abs(B(0, i)/B(rp+1, i));
                if (xx < min1)
                {
                    min1 = xx;
                    min1_index = i;
                }
            }
            for (int i = 0; i < m+1; i++)
            {
                if (i == rp+1)
                    continue;
                B.block(i, 0, 1, n+1) = B.block(i, 0, 1, n+1) - B(i, min1_index) / B(rp+1, min1_index) * B.block(rp+1, 0, 1, n+1);
            }
            B.block(rp+1, 0, 1, n+1) /= B(rp+1, min1_index);
            B_index(rp) = min1_index+1;
        }
        MatrixXd emmm(1, 1);
        emmm << 0;
        return emmm;
    }
private:
    MatrixXd A;
    MatrixXd b;
    MatrixXd c;
};
