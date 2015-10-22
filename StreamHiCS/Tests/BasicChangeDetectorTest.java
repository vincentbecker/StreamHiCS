import static org.junit.Assert.*;
import org.junit.BeforeClass;
import org.junit.Test;
import centroids.DensityChecker;
import centroids.FullSpaceContrastChecker;
import centroids.PHTChecker;
import contrast.Callback;
import contrast.CentroidContrast;
import streams.GaussianStream;
import weka.core.Instance;

public class BasicChangeDetectorTest {

	private static GaussianStream stream;
	private static CentroidContrast centroidContrast;
	private static Callback callback;
	private int numInstances = 10000;
	private static int counter = 0;
	private static double[][][] covarianceMatrices = {
			{ { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1 } },
			{ { 1, 0.2, 0.1, 0, 0 }, { 0.2, 1, 0.1, 0, 0 }, { 0.1, 0.1, 1, 0.1, 0.1 }, { 0, 0, 0.1, 1, 0.1 },
					{ 0, 0, 0, 0.1, 1 } },
			{ { 1, 0.4, 0.2, 0, 0 }, { 0.4, 1, 0.2, 0, 0 }, { 0.2, 0.2, 1, 0.2, 0.2 }, { 0, 0, 0.2, 1, 0.2 },
					{ 0, 0, 0, 0.2, 1 } },
			{ { 1, 0.6, 0.4, 0, 0 }, { 0.6, 1, 0.4, 0, 0 }, { 0.4, 0.4, 1, 0.4, 0.4 }, { 0, 0, 0.4, 1, 0.4 },
					{ 0, 0, 0, 0.4, 1 } },
			{ { 1, 0.8, 0.6, 0, 0 }, { 0.8, 1, 0.6, 0, 0 }, { 0.6, 0.6, 1, 0.6, 0.6 }, { 0, 0, 0.6, 1, 0.6 },
					{ 0, 0, 0, 0.6, 1 } },
			{ { 1, 0.9, 0.8, 0.2, 0.2 }, { 0.9, 1, 0.8, 0.2, 0.2 }, { 0.8, 0.8, 1, 0.8, 0.8 },
					{ 0.2, 0.2, 0.8, 1, 0.8 }, { 0.2, 0.2, 0.2, 0.8, 1 } } };

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		callback = new Callback() {

			@Override
			public void onAlarm() {
				System.out.println("Alarm");
				counter++;
			}

		};

		stream = new GaussianStream(covarianceMatrices[0]);
	}

	@Test
	public void test1() {
		System.out.println("DensityChecker: ");
		centroidContrast = new CentroidContrast(callback, 5, 20, 0.1, 0.005, 0.2, 1000, 0.1, 0.2,
				new DensityChecker(25, 1));

		assertTrue(carryOutTest());
	}

	@Test
	public void test2() {
		System.out.println("PHTChecker: ");
		centroidContrast = new CentroidContrast(callback, 5, 20, 0.1, 0.05, 0.2, 1000, 0.1, 0.2,
				new PHTChecker(5, 0.5, 1));

		assertTrue(carryOutTest());
	}

	@Test
	public void test3() {
		System.out.println("FullSpaceContrastChecker: ");
		FullSpaceContrastChecker fscc = new FullSpaceContrastChecker(5, null, 0.09);
		centroidContrast = new CentroidContrast(callback, 5, 50, 0.1, 0.005, 0.2, 1000, 0.1, 0.2, fscc);
		fscc.setContrastEvaluator(centroidContrast);

		assertTrue(carryOutTest());
	}

	private boolean carryOutTest() {
		boolean correct = true;
		int numberSamples = 0;
		for (int i = 0; i < covarianceMatrices.length; i++) {
			counter = 0;
			System.out.println("Iteration: " + i);
			stream.setCovarianceMatrix(covarianceMatrices[i]);
			numberSamples = 0;
			while (stream.hasMoreInstances() && numberSamples < numInstances) {
				Instance inst = stream.nextInstance();
				centroidContrast.add(inst);
				numberSamples++;
			}
			if (counter != 1) {
				correct = false;
			}
			System.out.println();
		}
		return correct;
	}
}
