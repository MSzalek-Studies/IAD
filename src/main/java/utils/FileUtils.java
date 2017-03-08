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

    public static Matrix loadMatrix(String filename) throws FileNotFoundException {
        File file = new File(filename);
        FileReader fileReader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        String[] lines = bufferedReader.lines().toArray(String[]::new);
        int rows = lines.length;
        int columns = lines[0].split(" ").length;

        Matrix matrix = new Basic2DMatrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            double[] array = Arrays.stream(lines[i].split(" ")).mapToDouble(Double::parseDouble).toArray();
            matrix.setRow(i, new BasicVector(array));
        }

        return matrix;
    }
}
