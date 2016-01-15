package ccd;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import environment.CSVReader;
import fullsystem.CorrelatedSubspacesChangeDetector;
import fullsystem.FullSpaceChangeDetector;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.trees.DecisionStump;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.streams.ConceptDriftStream;
import streams.GaussianStream;
import weka.core.Instance;

public class DriftTest {

	private ConceptDriftStream conceptDriftStream;
	private int numberSamples = 0;
	private final int numInstances = 20000;
	private final int numberOfDimensions = 5;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private CorrelatedSubspacesChangeDetector cscd;
	private FullSpaceChangeDetector refDetector;
	private static CSVReader csvReader;
	private static final String path = "Tests/CovarianceMatrices/";

	@Before
	public void setUp() throws Exception {

		csvReader = new CSVReader();
		numberSamples = 0;

		alpha = 0.15;
		epsilon = 0.1;
		threshold = 0.35;
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
		AbstractClassifier baseLearner = new HoeffdingAdaptiveTree();
		// AbstractClassifier baseLearner = new DecisionStump();
		// AbstractClassifier baseLearner = new NaiveBayes();
		baseLearner.prepareForUse();
		refDetector.baseLearnerOption.setCurrentObject(baseLearner);
		refDetector.prepareForUse();
	}

	/*
	@Test
	public void test1() {

		GaussianStream s1 = new GaussianStream(null, csvReader.read(path + "Test1.csv"), 1);

		// double[] mean2 = { 5, 5, 5, 5, 5 };
		double[] mean2 = null;
		GaussianStream s2 = new GaussianStream(mean2, csvReader.read(path + "Test1.csv"), 1.1);

		conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(s1);
		conceptDriftStream.driftstreamOption.setCurrentObject(s2);
		conceptDriftStream.positionOption.setValue(10000);
		conceptDriftStream.widthOption.setValue(1000);
		conceptDriftStream.prepareForUse();

		carryOutTest();
	}
	*/

	@Test
	public void test2() {

		GaussianStream s1 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1);

		// double[] mean2 = {5, 5, 5, 5, 5};
		double[] mean2 = null;
		GaussianStream s2 = new GaussianStream(mean2, csvReader.read(path + "Test2.csv"), 1.2);

		conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(s1);
		conceptDriftStream.driftstreamOption.setCurrentObject(s2);
		conceptDriftStream.positionOption.setValue(10000);
		conceptDriftStream.widthOption.setValue(1000);
		conceptDriftStream.prepareForUse();

		carryOutTest();
	}

	private void carryOutTest() {
		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			if (numberSamples % 1000 == 0) {
				/*
				System.out.println("Time: " + numberSamples);
				System.out.println("Number of microclusters: " + cscd.getNumberOfElements());
				
				SubspaceSet correlatedSubspaces = cscd.getCurrentlyCorrelatedSubspaces();
				System.out.println("Correlated: " + correlatedSubspaces);
				for (Subspace s : correlatedSubspaces.getSubspaces()) {
					System.out.print(s.getContrast() + ", ");
				}
				*/
				
				System.out.print("\n");
			}
			Instance inst = conceptDriftStream.nextInstance();
			cscd.trainOnInstance(inst);
			refDetector.trainOnInstance(inst);

			if (cscd.isWarningDetected()) {
				// System.out.println("cscd: WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				System.out.println("cscd: CHANGE at " + numberSamples);
			}

			if (refDetector.isWarningDetected()) {
				// System.out.println("refDetector: WARNING at " +
				// numberSamples);
			} else if (refDetector.isChangeDetected()) {
				System.out.println("refDetector: CHANGE at " + numberSamples);
			}

			numberSamples++;
		}

		fail("Not yet implemented");
	}
}
