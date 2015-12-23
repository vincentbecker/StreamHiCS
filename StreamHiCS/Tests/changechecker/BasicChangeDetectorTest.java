package changechecker;
import static org.junit.Assert.*;
import org.junit.BeforeClass;
import org.junit.Test;
import changechecker.FullSpaceContrastChecker;
import fullsystem.Callback;
import fullsystem.Contrast;
import moa.clusterers.clustree.ClusTree;
import streamdatastructures.MicroclusterAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.GaussianStream;
import weka.core.Instance;

public class BasicChangeDetectorTest {

	private static GaussianStream stream;
	private static Contrast contrast;
	private static Callback callback;
	private static int counter = 0;
	private int numInstances = 10000;
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

		stream = new GaussianStream(null, covarianceMatrices[0], 1);
		
		FullSpaceContrastChecker fscc = new FullSpaceContrastChecker(1000, 5, null, 0.2, 0.03);
		ClusTree mcs = new ClusTree();
		mcs.resetLearningImpl();
		SummarisationAdapter adapter = new MicroclusterAdapter(mcs);
		contrast = new Contrast(100, 0.1, adapter);
		fscc.setContrastEvaluator(contrast);
		fscc.setCallback(callback);

	}

	@Test
	public void test1() {
		assertTrue(carryOutTest(0));
	}

	@Test
	public void test2() {
		assertTrue(carryOutTest(1));
	}

	@Test
	public void test3() {
		assertTrue(carryOutTest(2));
	}

	@Test
	public void test4() {
		assertTrue(carryOutTest(3));
	}

	@Test
	public void test5() {
		assertTrue(carryOutTest(4));
	}

	@Test
	public void test6() {
		assertTrue(carryOutTest(5));
	}

	private boolean carryOutTest(int iteration) {
		int numberSamples = 0;
		counter = 0;
		System.out.println("Iteration: " + (iteration + 1));
		stream.setCovarianceMatrix(covarianceMatrices[iteration]);
		numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			contrast.add(inst);
			numberSamples++;
		}
		System.out.println();

		if (counter == 1) {
			return true;
		}

		return false;
	}
}
