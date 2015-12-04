package fullsystem;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import moa.classifiers.drift.SingleClassifierDrift;
import moa.streams.generators.RandomRBFGeneratorDrift;
import weka.core.Instance;

public class RBFDriftTest {

	private RandomRBFGeneratorDrift stream;
	private int numberSamples = 0;
	private final int numInstances = 10000;
	private final int numberOfDimensions = 10;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private CorrelatedSubspacesChangeDetector cscd;
	private FullSpaceChangeDetector refDetector;

	@Before
	public void setUp() throws Exception {
		stream = new RandomRBFGeneratorDrift();
		stream.speedChangeOption.setValue(1);
		stream.prepareForUse();

		//stream = new UncorrelatedStream();
		//stream.dimensionsOption.setValue(5);
		//stream.prepareForUse();
		
		alpha = 0.15;
		epsilon = 0.15;
		threshold = 0.25;
		cutoff = 8;
		pruningDifference = 0.15;

		cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions);
		cscd.alphaOption.setValue(alpha);
		cscd.epsilonOption.setValue(epsilon);
		cscd.thresholdOption.setValue(threshold);
		cscd.cutoffOption.setValue(cutoff);
		cscd.pruningDifferenceOption.setValue(pruningDifference);
		cscd.prepareForUse();
		
		refDetector = new FullSpaceChangeDetector();
		refDetector.prepareForUse();
	}

	@Test
	public void test() {
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			if (numberSamples % 1000 == 0) {
				//System.out.println("Time: " + numberSamples);
				//System.out.println("Number of microclusters: " + cscd.getNumberOfElements());
				//System.out.println("sh2: " + sh2.toString());
			}
			Instance inst = stream.nextInstance();
			cscd.trainOnInstance(inst);
			refDetector.trainOnInstance(inst);
			/*
			DenseInstance newInst = new DenseInstance(numberOfDimensions);
			for(int i = 0; i < numberOfDimensions; i++){
				newInst.setValue(i, inst.value(i));
			}
			*/
			
			if (cscd.isWarningDetected()) {
				System.out.println("cscd: WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				System.out.println("cscd: CHANGE at " + numberSamples);
			}
			
			if (refDetector.isWarningDetected()) {
				System.out.println("refDetector: WARNING at " + numberSamples);
			} else if (refDetector.isChangeDetected()) {
				System.out.println("refDetector: CHANGE at " + numberSamples);
			}
			
			numberSamples++;
		}

		fail("Not yet implemented");
	}

}
