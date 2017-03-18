import models.NeuralNetwork;
import org.la4j.Matrix;
import utils.FileUtils;

import java.io.FileNotFoundException;

/**
 * Created by marcinus on 18.03.17.
 */
public class Transofrmation {

    public void performTransformation() {
        try {
            Matrix[] matrices = new FileUtils().loadDataFromTwoFiles("transformation.txt", "transformation.txt");
            NeuralNetwork nn = new NeuralNetwork(matrices[0], matrices[1], false, new int[]{5, 5, 5});
            nn.train(500, 0.001);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
