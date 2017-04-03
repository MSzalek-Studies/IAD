import models.NeuralNetwork;
import org.jfree.data.xy.XYSeries;
import org.la4j.Matrix;
import utils.ErrorChart;
import utils.FileUtils;

import java.io.FileNotFoundException;

/**
 * Created by marcinus on 18.03.17.
 */
public class Transofrmation {

    public void performTransformation() {
        try {
            Matrix[] matrices = new FileUtils().loadDataFromTwoFiles("transformation.txt", "transformation.txt");
            ErrorChart errorChart = new ErrorChart();

            NeuralNetwork nn = new NeuralNetwork(matrices[0], matrices[1], true, new int[]{1}, 1, 0.9);
            XYSeries errorSeries = nn.train(5000, 0.001);
            errorSeries.setKey("Przebieg dla jednego neuronu z biasem");
            errorChart.addSeries(errorSeries);

            nn = new NeuralNetwork(matrices[0], matrices[1], true, new int[]{1}, 1, 0.9);
            errorSeries = nn.train(5000, 0.001);
            errorSeries.setKey("Przebieg dla jednego neuronu bez biasu");
            errorChart.addSeries(errorSeries);

            errorChart.generateChart();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
