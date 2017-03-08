import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import utils.MatrixUtils;

import java.io.FileNotFoundException;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Main {

    public static void main(String... args) {
        Matrix matrix = new Basic2DMatrix(2, 2);
        matrix.set(0, 0, 5);
        matrix.set(1, 1, -20);
        matrix = MatrixUtils.sigmoid(matrix);
        matrix = MatrixUtils.addBiasColumn(matrix);
        try {
            new Ex1().doMagic();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
