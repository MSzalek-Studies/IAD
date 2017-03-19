package activator;

import org.la4j.Matrix;

/**
 * Created by marcinus on 19.03.17.
 */
public class SigmoidActivator implements Activator {

    @Override
    public Matrix activate(Matrix matrix) {
        matrix = matrix.transform((i, i1, v) -> 1 / (1 + Math.exp(-v)));
        return matrix.transform((i, j, v) -> Double.isNaN(v) ? 0 : v);
    }
}
