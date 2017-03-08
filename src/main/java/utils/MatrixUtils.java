package utils;

import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.matrix.functor.MatrixFunction;

import java.util.Random;

/**
 * Created by marcinus on 08.03.17.
 */
public class MatrixUtils {

    public static Matrix sigmoid(Matrix matrix) {
        return matrix.transform((i, i1, v) -> 1 / (1 + Math.exp(-v)));
    }

    public static Matrix addBiasColumn(Matrix matrix) {
        Matrix tempMatrix = new Basic2DMatrix(matrix.rows(), matrix.columns() + 1);
        tempMatrix.setAll(1);
        matrix = tempMatrix.insert(matrix, 0, 1, matrix.rows(), matrix.columns());
        return matrix;
    }

    public static Matrix randomlyInitWeights(int rows, int columns) {
        double epsilonInit = Math.sqrt(6) / (Math.sqrt(rows) + Math.sqrt(columns));
        Matrix matrix = new Basic2DMatrix(rows, columns);
        matrix = matrix.transform(new MatrixFunction() {
            @Override
            public double evaluate(int i, int j, double value) {
                return new Random().nextDouble() * 2 * epsilonInit - epsilonInit;
            }
        });
        return matrix;
    }
}
