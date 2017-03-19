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
            NeuralNetwork nn = new NeuralNetwork(matrices[0], matrices[1], true, new int[]{5, 5, 5});
            ErrorChart errorChart = new ErrorChart();
            XYSeries errorSeries = nn.train(500, 0.001);
            errorChart.addSeries(errorSeries);
            errorChart.generateChart();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
