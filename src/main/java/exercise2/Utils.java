package exercise2;

import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;

import java.util.Random;

/**
 * Created by marcinus on 26.04.17.
 */
public class Utils {

    public Matrix initNeurons(Matrix inputMatrix, int k) {
        Matrix neurons = new Basic2DMatrix(k, 2);
        double minXArg = inputMatrix.getColumn(0).min();
        double maxXArg = inputMatrix.getColumn(0).max();
        double minYArg = inputMatrix.getColumn(1).min();
        double maxYArg = inputMatrix.getColumn(1).max();
        for (int i = 0; i < k; i++) {
            double x = new Random().nextDouble() * (maxXArg - minXArg) + minXArg;
            double y = new Random().nextDouble() * (maxYArg - minYArg) + minYArg;
            neurons.set(i, 0, x);
            neurons.set(i, 1, y);
        }
        return neurons;
    }
}
