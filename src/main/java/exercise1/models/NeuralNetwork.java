package exercise1.models;

import exercise1.activator.Activator;
import exercise1.activator.SigmoidActivator;
import org.jfree.data.xy.XYSeries;
import org.la4j.Matrix;
import org.la4j.vector.dense.BasicVector;

import java.util.Arrays;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class NeuralNetwork {

    private int inputSize;
    private int outputSize;
    private boolean includeBias;
    private double learningRate;
    private double momentum;

    protected InputLayer inputLayer;
    private Layer[] hiddenLayers;
    private OutputLayer outputLayer;

    private Activator activator;
    private Activator outputActivator;

    public NeuralNetwork(int inputSize, int outputSize,
                         boolean includeBias, int[] hiddenLayerSizes, double learningRate, double momentum) {
        this(inputSize, outputSize, includeBias, hiddenLayerSizes, new SigmoidActivator(), learningRate, momentum);
    }

    public NeuralNetwork(int inputSize, int outputSize,
                         boolean includeBias, int[] hiddenLayerSizes, Activator outputActivator, double learningRate, double momentum) {
        this(inputSize, outputSize, includeBias, hiddenLayerSizes, outputActivator, new SigmoidActivator(), learningRate, momentum);
    }

    public NeuralNetwork(int inputSize, int outputSize,
                         boolean includeBias, int[] hiddenLayerSizes,
                         Activator outputActivator, Activator hiddenActivator,
                         double learningRate, double momentum) {
        this.outputActivator = outputActivator;
        this.activator = hiddenActivator;
        this.momentum = momentum;
        this.learningRate = learningRate;

        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.includeBias = includeBias;

        initLayers(hiddenLayerSizes);
    }

    public XYSeries train(Matrix inputMatrix, Matrix expectedResults, int maxIterations, double desiredError) {
        int numExamples = inputMatrix.rows();
        inputLayer.setInput(inputMatrix);
        XYSeries series = new XYSeries("Seria");
        int iteration = 0;
        double error;
        do {
            forwardPropagateNetwork();
            calculateErrors(expectedResults);
            backwardPropagate();
            gradientDescent();

            iteration++;
            error = outputLayer.cost(expectedResults, numExamples);
            series.add(iteration, error);
        }
        while (iteration < maxIterations && error > desiredError);

        //showResults(expectedResults);
        return series;
    }

    /**
     * @param input mxn matrix
     * @return
     */
    public Matrix predict(Matrix input) {
        inputLayer.setInput(input);
        forwardPropagateNetwork();
        return outputLayer.getActivationValues();
    }

    public double test(Matrix input, Matrix expectedValues) {
        inputLayer.setInput(input);
        forwardPropagateNetwork();
        return outputLayer.cost(expectedValues, input.rows());
    }

    private void gradientDescent() {
        for (Layer layer : hiddenLayers) {
            layer.gradientDescent();
        }
        outputLayer.gradientDescent();
    }

    private void backwardPropagate() {
        if (hiddenLayers.length > 0) {
            outputLayer.propagateBackward(hiddenLayers[hiddenLayers.length - 1].getActivationValues());
            for (int i = hiddenLayers.length - 1; i > 0; i--) {
                hiddenLayers[i].propagateBackward(hiddenLayers[i - 1].getActivationValues());
            }
            hiddenLayers[0].propagateBackward(inputLayer.getActivationValues());
        } else {
            outputLayer.propagateBackward(inputLayer.getActivationValues());
        }
    }

    private void calculateErrors(Matrix expectedResults) {
        outputLayer.calculateErrors(expectedResults);
        if (hiddenLayers.length > 0) {
            hiddenLayers[hiddenLayers.length - 1].calculateErrors(outputLayer);
            for (int i = hiddenLayers.length - 2; i >= 0; i--) {
                hiddenLayers[i].calculateErrors(hiddenLayers[i + 1]);
            }
        }
    }

    private void showResults(Matrix expectedResults) {
        forwardPropagateNetwork();

        System.out.println("\n\n================================\n\n");
        for (int i = 0; i < expectedResults.rows(); i++) {
            System.out.println("expected: " + Arrays.toString(((BasicVector) expectedResults.getRow(i)).toArray()));
            System.out.println("output: " + Arrays.toString(((BasicVector) outputLayer.getActivationValues().getRow(i)).toArray()) + "\n");
        }
    }

    private void forwardPropagateNetwork() {
        if (hiddenLayers.length > 0) {
            hiddenLayers[0].forwardPropagate(inputLayer);
            for (int i = 1; i < hiddenLayers.length; i++) {
                hiddenLayers[i].forwardPropagate(hiddenLayers[i - 1]);
            }
            outputLayer.forwardPropagate(hiddenLayers[hiddenLayers.length - 1]);
        } else {
            outputLayer.forwardPropagate(inputLayer);
        }
    }

    //
    // ============= INIT ================
    //
    private void initLayers(int[] hiddenLayerSizes) {
        hiddenLayers = new Layer[hiddenLayerSizes.length];
        inputLayer = new InputLayer(inputSize, includeBias, learningRate, momentum);
        if (hiddenLayerSizes.length > 0) {
            initLayersWithHidden(hiddenLayerSizes);
        } else {
            initOutputLayer(inputSize);
        }
    }

    private void initLayersWithHidden(int[] hiddenLayerSizes) {
        initHiddenLayers(hiddenLayerSizes);
        int lastHiddenLayerSize = hiddenLayerSizes[hiddenLayerSizes.length - 1];
        initOutputLayer(lastHiddenLayerSize);
    }

    private void initOutputLayer(int numInputsWithoutBias) {
        outputLayer = new OutputLayer(
                numInputsWithoutBias + (includeBias ? 1 : 0),
                outputSize,
                outputActivator,
                learningRate,
                momentum);
    }

    private void initHiddenLayers(int[] hiddenLayerSizes) {
        Layer firstHiddenLayer = new Layer(
                inputSize + (includeBias ? 1 : 0),
                hiddenLayerSizes[0],
                includeBias,
                activator,
                learningRate,
                momentum);
        hiddenLayers[0] = firstHiddenLayer;
        for (int i = 1; i < hiddenLayerSizes.length; i++) {
            Layer layer = new Layer(
                    hiddenLayerSizes[i - 1] + (includeBias ? 1 : 0),
                    hiddenLayerSizes[i],
                    includeBias,
                    activator,
                    learningRate,
                    momentum);
            hiddenLayers[i] = layer;
        }
    }

}
