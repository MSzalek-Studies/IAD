package utils;

import org.la4j.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.dense.BasicVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;

/**
 * Created by marcinus on 08.03.17.
 */
public class FileUtils {

    public Matrix loadMatrix(String filename, String splitSymbol) throws FileNotFoundException {
        File file = new File(filename);
        FileReader fileReader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        String[] lines = bufferedReader.lines().toArray(String[]::new);
        int rows = lines.length;
        int columns = lines[0].split(splitSymbol).length;

        Matrix matrix = new Basic2DMatrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            double[] array = Arrays.stream(lines[i].split(splitSymbol)).mapToDouble(Double::parseDouble).toArray();
            matrix.setRow(i, new BasicVector(array));
        }

        return matrix;
    }

    /**
     * @param inputFileName - file with inputData that also contains expectedResults in last column
     * @return firstElem - inputMatrix,
     * secondElem - expectedResultsMatrix
     * size = 2
     */
    public Matrix[] loadDataFromSingleFileSupervised(String inputFileName) throws FileNotFoundException {
        Matrix inputMatrix = loadMatrix(inputFileName, " ");
        Matrix expectedResults = new Basic2DMatrix(inputMatrix.rows(), 1);
        expectedResults.setColumn(0, inputMatrix.getColumn(inputMatrix.columns() - 1));
        inputMatrix = inputMatrix.removeColumn(inputMatrix.columns() - 1);
        return new Matrix[]{inputMatrix, expectedResults};
    }

    public Matrix loadDataFromSingleFileUnsupervised(String inputFileName) throws FileNotFoundException {
        return loadMatrix(inputFileName, ",");
    }

    /**
     * @param inputFileName  - file with inputData
     * @param valuesFileName - file with expectedResults
     * @return firstElem - inputMatrix,
     * secondElem - expectedResultsMatrix
     * size = 2
     */
    public Matrix[] loadDataFromTwoFiles(String inputFileName, String valuesFileName) throws FileNotFoundException {
        Matrix inputMatrix = loadMatrix(inputFileName, " ");
        Matrix expectedResults = loadMatrix(valuesFileName, " ");
        return new Matrix[]{inputMatrix, expectedResults};
    }
}
