package ccd;

import java.util.ArrayList;

import org.junit.Before;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.CSVReader;
import environment.Evaluator;
import environment.Stopwatch;
import fullsystem.Contrast;
import fullsystem.CorrelatedSubspacesChangeDetector;
import fullsystem.FullSpaceChangeDetector;
import fullsystem.StreamHiCS;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.ConceptDriftStream;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.GaussianStream;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;
import weka.core.Utils;

public class DriftTest {

	private ConceptDriftStream conceptDriftStream;
	private int numberSamples = 0;
	private final int numInstances = 20000;
	private final int numberOfDimensions = 5;
	private int m = 50;
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
		m = 50;
		alpha = 0.15;
		epsilon = 0.1;
		threshold = 0.35;
		cutoff = 8;
		pruningDifference = 0.15;

		int horizon = 2000;

		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(horizon);
		// mcs.horizonOption.setValue(20000);
		mcs.prepareForUse();

		// StreamHiCS
		SummarisationAdapter adapter = new MicroclusteringAdapter(mcs);
		Contrast contrastEvaluator = new Contrast(m, alpha, adapter);
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		CorrelationSummary correlationSummary = new CorrelationSummary(numberOfDimensions, horizon);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, contrastEvaluator,
				correlationSummary);
		Stopwatch stopwatch = new Stopwatch();
		StreamHiCS streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator,
				subspaceBuilder, changeChecker, null, correlationSummary, stopwatch);
		changeChecker.setCallback(streamHiCS);

		cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions, streamHiCS);
		cscd.prepareForUse();
		streamHiCS.setCallback(cscd);

		refDetector = new FullSpaceChangeDetector();
		AbstractClassifier baseLearner = new HoeffdingTree();
		// AbstractClassifier baseLearner = new DecisionStump();
		// AbstractClassifier baseLearner = new NaiveBayes();
		baseLearner.prepareForUse();
		refDetector.baseLearnerOption.setCurrentObject(baseLearner);
		refDetector.prepareForUse();
	}

	/*
	 * @Test public void test1() {
	 * 
	 * GaussianStream s1 = new GaussianStream(null, csvReader.read(path +
	 * "Test1.csv"), 1);
	 * 
	 * // double[] mean2 = { 5, 5, 5, 5, 5 }; double[] mean2 = null;
	 * GaussianStream s2 = new GaussianStream(mean2, csvReader.read(path +
	 * "Test1.csv"), 1.1);
	 * 
	 * conceptDriftStream = new ConceptDriftStream();
	 * conceptDriftStream.streamOption.setCurrentObject(s1);
	 * conceptDriftStream.driftstreamOption.setCurrentObject(s2);
	 * conceptDriftStream.positionOption.setValue(10000);
	 * conceptDriftStream.widthOption.setValue(1000);
	 * conceptDriftStream.prepareForUse();
	 * 
	 * carryOutTest(); }
	 */

	@Test
	public void test2() {

		GaussianStream s1 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 0.675);

		// double[] mean2 = {5, 5, 5, 5, 5};
		double[] mean2 = null;
		GaussianStream s2 = new GaussianStream(mean2, csvReader.read(path + "Test2.csv"), 1);

		conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(s1);
		conceptDriftStream.driftstreamOption.setCurrentObject(s2);
		conceptDriftStream.positionOption.setValue(10000);
		conceptDriftStream.widthOption.setValue(1000);
		conceptDriftStream.prepareForUse();

		ArrayList<Double> trueChanges = new ArrayList<Double>();
		trueChanges.add(9500.0);

		carryOutTest(trueChanges);
	}

	private void carryOutTest(ArrayList<Double> trueChanges) {
		ArrayList<Double> cscdDetectedChanges = new ArrayList<Double>();
		ArrayList<Double> refDetectedChanges = new ArrayList<Double>();

		int numberCorrectCSCD = 0;
		int numberCorrectREF = 0;

		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			/*
			if (numberSamples % 1000 == 0) {

				System.out.println("Time: " + numberSamples);
				System.out.println("Number of microclusters: " + cscd.getNumberOfElements());

				SubspaceSet correlatedSubspaces = cscd.getCurrentlyCorrelatedSubspaces();
				System.out.println("Correlated: " + correlatedSubspaces);
				for (Subspace s : correlatedSubspaces.getSubspaces()) {
					System.out.print(s.getContrast() + ", ");
				}

				System.out.print("\n");
			}
			*/
			Instance inst = conceptDriftStream.nextInstance();

			// For accuracy
			int trueClass = (int) inst.classValue();
			if (trueClass == cscd.getClassPrediction(inst)) {
				numberCorrectCSCD++;
			}
			if (trueClass == Utils.maxIndex(refDetector.getVotesForInstance(inst))) {
				numberCorrectREF++;
			}

			cscd.trainOnInstance(inst);
			refDetector.trainOnInstance(inst);

			if (cscd.isWarningDetected()) {
				// System.out.println("cscd: WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				System.out.println("cscd: CHANGE at " + numberSamples);
				cscdDetectedChanges.add((double) numberSamples);
			}

			if (refDetector.isWarningDetected()) {
				// System.out.println("refDetector: WARNING at " +
				// numberSamples);
			} else if (refDetector.isChangeDetected()) {
				System.out.println("refDetector: CHANGE at " + numberSamples);
				refDetectedChanges.add((double) numberSamples);
			}

			numberSamples++;
		}

		double[] cscdPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, cscdDetectedChanges, 1000, 
				numInstances);
		double[] refPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, refDetectedChanges, 1000, 
				numInstances);

		double cscdAccuracy = ((double) numberCorrectCSCD) / numInstances;
		double refAccuracy = ((double) numberCorrectREF) / numInstances;
		String cscdMeasures = "CSCD";
		String refMeasures = "REF";
		for (int i = 0; i < 4; i++) {
			cscdMeasures += ", " + cscdPerformanceMeasures[i];
			refMeasures += ", " + refPerformanceMeasures[i];
		}
		cscdMeasures += "," + cscdAccuracy;
		refMeasures += "," + refAccuracy;

		System.out.println(cscdMeasures);
		System.out.println(refMeasures);
	}
}
