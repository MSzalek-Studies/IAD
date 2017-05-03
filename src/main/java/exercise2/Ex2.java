package exercise2;

import org.la4j.Matrix;
import utils.FileUtils;

import java.io.FileNotFoundException;

/**
 * Created by marcinus on 20.04.17.
 */
public class Ex2 {
    public static void main(String[] args) {
        try {
            Matrix input = new FileUtils().loadDataFromSingleFileUnsupervised("attract.txt");
            Option option = Option.NGAS;
            switch (option) {
                case KMEANS: {
                    KMeans kMeans = new KMeans(input, 200);
                    kMeans.perform(100);
                    break;
                }
                case KOHONEN: {
                    Kohonen kohonen = new Kohonen(input, 100);
                    kohonen.perform(100, 0.1, 0.01);
                    break;
                }
                case NGAS: {
                    NeuralGas neuralGas = new NeuralGas(input, 200);
                    neuralGas.perform(100, 0.1, 1);
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    enum Option {
        KMEANS, KOHONEN, NGAS;
    }

}
