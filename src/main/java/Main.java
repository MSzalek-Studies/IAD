import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.FileNotFoundException;

/**
 * Created by szale_000 on 2017-03-07.
 */
public class Main {

    public static void main(String... args) {
        try {
            new Ex1().doMagic();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
