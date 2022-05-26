#pragma once
#include <eigen3/Eigen/Dense>

using namespace Eigen;

class SIMPLEX
{
public:
    SIMPLEX(MatrixXd A1, MatrixXd b1, MatrixXd c1)
    {
        A = A1;
        b = b1;
        c = c1;
    }
    MatrixXd solve()
    {
        // start
        int m = A.rows();
        int n = A.cols();
        MatrixXd B(m, m);
        MatrixXd cB(m, 1);
        MatrixXd N(m, n-m);
        MatrixXd cNB(n-m, 1);
        VectorXd Basic_Index(m);
        VectorXd NoBasic_Index(n-m);
        for (int i = 0; i < m; i++)
        {
            Basic_Index(i) = i;
            B.col(i) = A.col(i);
            cB(i) = c(i);
        }
        for (int i = 0; i < n-m; i++)
        {
            NoBasic_Index(i) = m+i;
            N.col(i) = A.col(m+i);
            cNB(i) = c(m+i);
        }
        MatrixXd xB = B.inverse() * b;
        // if best
        while (true)
        {
            MatrixXd w = (B.adjoint()).inverse() * cB;
            MatrixXd z = w.adjoint() * N;
            int p = -1; // the index to get into basic
            for (int j = 0; j < n-m; j++)
            {
                if(cNB(j) - z(j) < 0)
                {
                    p = j;
                    break;
                }
            }
            if (p == -1)
            {
                //return xB;
                MatrixXd X(1, n);
                for(int j = 0; j < m; j++)
                    X(Basic_Index(j)) = xB(j);
                for(int j = 0; j < n-m; j++)
                    X(NoBasic_Index(j)) = 0;

                return X;
            }
            MatrixXd yp = B.inverse() * N.col(p);
            bool ifok = false;
            for (int j = 0; j < m; j++)
            {
                if (yp(j) > 0)
                {
                    ifok = true;
                    break;
                }
            }
            if (!ifok) // no result
            {
                MatrixXd emmm(1, 1);
                emmm << 0;
                return emmm;
            }
            int r, c;
            double delta = (xB.cwiseQuotient(yp)).minCoeff(&r, &c);
            xB = xB - delta * yp;
            xB(r) = delta;
            double b_o_index = Basic_Index(r);
            Basic_Index(r) = NoBasic_Index(p);
            NoBasic_Index(p) = b_o_index;
            double b_o_c = cB(r);
            cB(r) = cNB(p);
            cNB(p) = b_o_c;
            MatrixXd B_o = B.col(r);
            B.col(r) = N.col(p);
            N.col(p) = B_o;
        }
    }
private:
    MatrixXd A;
    MatrixXd b;
    MatrixXd c;
};
