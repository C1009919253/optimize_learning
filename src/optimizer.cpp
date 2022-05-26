#include "gurobi_c++.h"
#include <eigen3/Eigen/Dense>
#include "optimize_learning/SIMPLEX.hpp"

using namespace std;
#if 0
int
main(int   argc,
     char *argv[])
{
  try {

    // Create an environment
    GRBEnv env = GRBEnv(true);
    env.set("LogFile", "mip1.log");
    env.start();

    // Create an empty model
    GRBModel model = GRBModel(env);

    // Create variables
    /*GRBVar x = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "x");
    GRBVar y = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS, "y");
    GRBVar z = model.addVar(0.0, 1.5, 0.0, GRB_CONTINUOUS, "z");*/

    /*// Set objective: maximize x + y + 2 z
    model.setObjective(x + y + 2 * z, GRB_MAXIMIZE);

    // Add constraint: x + 2 y + 3 z <= 4
    model.addConstr(x + 2 * y + 3 * z <= 4, "c0");

    // Add constraint: x + y >= 1
    model.addConstr(x + y <= 1, "c1");*/

    GRBVar x = model.addVar(-100, 100, 0.0, GRB_CONTINUOUS, "x");
    GRBVar y = model.addVar(-100, 100, 0.0, GRB_CONTINUOUS, "y");
    GRBVar theta = model.addVar(-3.15, 3.15, 0.0, GRB_CONTINUOUS, "theta");
    GRBVar v = model.addVar(-2, 2, 0.0, GRB_CONTINUOUS, "v");
    GRBVar w = model.addVar(-1, 1, 0.0, GRB_CONTINUOUS, "w");

    model.setObjective(x + y + 2 * theta, GRB_MINIMIZE);



    // Optimize model
    model.optimize();

    cout << x.get(GRB_StringAttr_VarName) << " "
         << x.get(GRB_DoubleAttr_X) << endl;
    cout << y.get(GRB_StringAttr_VarName) << " "
         << y.get(GRB_DoubleAttr_X) << endl;
    cout << theta.get(GRB_StringAttr_VarName) << " "
         << theta.get(GRB_DoubleAttr_X) << endl;

    cout << "Obj: " << model.get(GRB_DoubleAttr_ObjVal) << endl;

  } catch(GRBException e) {
    cout << "Error code = " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch(...) {
    cout << "Exception during optimization" << endl;
  }

  return 0;
}
#endif

void pri(Eigen::MatrixXd m)
{
    std::cout << m << std::endl;
}

int main()
{
    Eigen::MatrixXd A(3, 5);
    //A << 2, 1, 1, 0, 1, 2, 0, 1;
    A << 0, 5, 1, 0, 0,
         6, 2, 0, 1, 0,
         1, 1, 0, 0, 1;
    Eigen::MatrixXd b(3, 1);
    //b << 6, 6;
    b << 15, 24, 5;
    Eigen::MatrixXd c(1, 5);
    //c << -3, -2, 0, 0;
    c << -2, -1, 0, 0, 0;
    SIMPLEX test(A, b, c);
    pri(test.solve());
}
