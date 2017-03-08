import com.sun.org.apache.regexp.internal.RE;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Ex1 {

    public void doMagic() throws FileNotFoundException {
        RealMatrix inputs = loadMatrix("transformation.txt");
        int examples = inputs.getRowDimension();

        int inputSize = 4;
        InputLayer inputLayer = new InputLayer(inputSize);

        int hiddenSize = 2;
        Layer hiddenLayer = new Layer(inputSize,hiddenSize);

        int outputSize = 4;
        OutputLayer outputLayer = new OutputLayer(hiddenSize, outputSize);

        for (int it=0; it<5000; it++) {
            double cost = 0;
            for (int i = 0; i < examples; i++) {
                inputLayer.setInput(inputs.getRow(i));
                hiddenLayer.forwardPropagate(inputLayer);
                outputLayer.forwardPropagate(hiddenLayer);

                RealVector expectedOutput = inputs.getRowVector(i);
                outputLayer.calculateErrors(expectedOutput);
                hiddenLayer.calculateErrors(outputLayer);

                outputLayer.propagateBackward(hiddenLayer.activationValues);
                hiddenLayer.propagateBackward(inputLayer.activationValues);
                cost += outputLayer.cost(expectedOutput);


            }
            hiddenLayer.gradientDescent(examples);
            outputLayer.gradientDescent(examples);

            if (it%50==0) {
                System.out.println(it + ": " + cost / examples);
            }
            hiddenLayer.clearExceptWeights();
            outputLayer.clearExceptWeights();
        }
        System.out.println("\n\n================================\n\n");
        for (int i = 0; i < examples; i++) {
            inputLayer.setInput(inputs.getRow(i));
            hiddenLayer.forwardPropagate(inputLayer);
            outputLayer.forwardPropagate(hiddenLayer);

            System.out.println("input: "+Arrays.toString(inputs.getRow(i)));
            System.out.println("output: "+Arrays.toString(outputLayer.getActivationValues().toArray()));
        }
    }

    private RealMatrix loadMatrix(String filename) throws FileNotFoundException {
        File file = new File(filename);
        FileReader fileReader = new FileReader(file);
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        String[] lines = bufferedReader.lines().toArray(String[]::new);
        int rows = lines.length;
        int columns = lines[0].split(" ").length;

        RealMatrix matrix = MatrixUtils.createRealMatrix(rows, columns);

        for (int i = 0; i < rows; i++) {
            double[] array = Arrays.stream(lines[i].split(" ")).mapToDouble(Double::parseDouble).toArray();
            matrix.setRow(i, array);
        }

        return matrix;
    }

}
