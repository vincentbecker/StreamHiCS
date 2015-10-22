import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import centroids.ChangeChecker;
import centroids.TimeCountChecker;
import contrast.CentroidContrast;
import contrast.Contrast;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.RandomRBFGeneratorDrift;
import weka.core.Instance;

public class RBFDriftTest {

	private RandomRBFGenerator stream;
	private StreamHiCS streamHiCS;
	private int numberSamples = 0;
	private final int numInstances = 10000;

	@Before
	public void setUp() throws Exception {
		stream = new RandomRBFGenerator();
		stream.prepareForUse();
		ChangeChecker changeChecker = new TimeCountChecker();
		Contrast contrastEvaluator = new CentroidContrast(null, 10, 50, 0.1, 0.005, 0.2, 1000, 0.1, 0.1, changeChecker);
		streamHiCS = new StreamHiCS(10, 0.05, 0.2, 8, 0.1, contrastEvaluator);
		contrastEvaluator.setCallback(streamHiCS);
	}

	@Test
	public void test() {
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		fail("Not yet implemented");
	}

}
