import com.sun.istack.internal.Nullable;
import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import utils.FileUtils;

import java.io.FileNotFoundException;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Main {

    public static void main(String... args) {
        try {
            Matrix[] matrices = loadData("transformation.txt", "transformation.txt");
            NeuralNetwork nn = new NeuralNetwork(matrices[0], matrices[1], 3, true);
            nn.train(500);
            Matrix test = new Basic2DMatrix(new double[][]{{0, 0, 1, 0}});
            Matrix result = nn.predict(test);
            System.out.print(result.toString());
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    /**
     * @param inputFileName
     * @param valuesFileName - can be null if first matrix contains values in last column
     * @return firstElem - inputMatrix,
     * secondElem - expectedResultsMatrix
     * size = 2
     */
    private static Matrix[] loadData(String inputFileName, @Nullable String valuesFileName) throws FileNotFoundException {
        Matrix inputMatrix = FileUtils.loadMatrix(inputFileName);
        Matrix expectedResults;
        if (valuesFileName == null) {
            expectedResults = new Basic2DMatrix(inputMatrix.rows(), 1);
            expectedResults = expectedResults.insertColumn(0, inputMatrix.getColumn(inputMatrix.columns() - 1));
            inputMatrix = inputMatrix.removeColumn(inputMatrix.columns() - 1);
        } else {
            expectedResults = FileUtils.loadMatrix(valuesFileName);
        }
        return new Matrix[]{inputMatrix, expectedResults};
    }
}
