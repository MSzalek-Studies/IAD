import models.InputLayer;
import models.Layer;
import models.OutputLayer;
import org.la4j.Matrix;
import org.la4j.vector.dense.BasicVector;
import utils.FileUtils;

import java.io.FileNotFoundException;
import java.util.Arrays;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Ex1 {

    public void doMagic() throws FileNotFoundException {
        Matrix inputs = FileUtils.loadMatrix("transformation.txt");
        Matrix expectedResults = FileUtils.loadMatrix("transformation.txt");
        int examples = inputs.rows();
        boolean includeBias = true;

        int inputSize = inputs.getRow(0).length();
        int hiddenSize = 2;
        int outputSize = expectedResults.getRow(0).length();

        InputLayer inputLayer = new InputLayer(inputSize, examples, includeBias);
        Layer hiddenLayer = new Layer(inputSize + (includeBias ? 1 : 0), hiddenSize, examples, includeBias);
        OutputLayer outputLayer = new OutputLayer(hiddenSize + (includeBias ? 1 : 0), outputSize, examples);

        for (int it = 0; it < 500; it++) {
            double cost = 0;
            inputLayer.setInput(inputs);
            hiddenLayer.forwardPropagate(inputLayer);
            outputLayer.forwardPropagate(hiddenLayer);
            cost = outputLayer.cost(expectedResults);

            outputLayer.calculateErrors(expectedResults);
            hiddenLayer.calculateErrors(outputLayer);

            outputLayer.propagateBackward(hiddenLayer.getActivationValues());
            hiddenLayer.propagateBackward(inputLayer.getActivationValues());

            hiddenLayer.gradientDescent();
            outputLayer.gradientDescent();

            if (it%50==0) {
                System.out.println(it + ": " + cost / examples);
            }
        }
        System.out.println("\n\n================================\n\n");
        for (int i = 0; i < examples; i++) {
            //inputLayer.setInput(inputs.getRow(i));
            hiddenLayer.forwardPropagate(inputLayer);
            outputLayer.forwardPropagate(hiddenLayer);

            System.out.println("input: " + Arrays.toString(((BasicVector) inputs.getRow(i)).toArray()));
            System.out.println("input: " + Arrays.toString(((BasicVector) expectedResults.getRow(i)).toArray()));
            System.out.println("output: " + Arrays.toString(((BasicVector) outputLayer.getActivationValues().getRow(i)).toArray()) + "\n");
        }
    }

}
