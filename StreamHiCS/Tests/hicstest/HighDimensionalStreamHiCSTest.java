package hicstest;

import static org.junit.Assert.*;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import contrast.Contrast;
import contrast.MicroclusterContrast;
import environment.CSVReader;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import fullsystem.Callback;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class HighDimensionalStreamHiCSTest {

	private GaussianStream stream;
	private StreamHiCS streamHiCS;
	private Contrast contrastEvaluator;
	private final int numInstances = 10000;
	private final int m = 50;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			System.out.println("StreamHiCS: onAlarm()");
		}
	};

	@Test
	public void subspaceTest1() {
		String testName = "Test1";
		double[][] covarianceMatrix = CovarianceMatrixGenerator.generateCovarianceMatrix(60, 10, 0.9);
		// No correlated subspaces should have been found, so the correctResult
		// is empty.
		SubspaceSet correctResult = new SubspaceSet();
		correctResult.addSubspace(new Subspace(0, 1, 2, 3, 4));
		System.out.println(testName);
		assertTrue(carryOutSubspaceTest(covarianceMatrix, correctResult));
	}

	private boolean carryOutSubspaceTest(double[][] covarianceMatrix, SubspaceSet correctResult) {
		stream = new GaussianStream(covarianceMatrix);

		alpha = 0.2;
		epsilon = 0;
		threshold = 0.4;
		cutoff = 8;
		pruningDifference = 0.15;

		ClusTree mcs = new ClusTree();
		mcs.resetLearningImpl();
		contrastEvaluator = new MicroclusterContrast(m, alpha, mcs);

		SubspaceBuilder subspaceBuilder = new AprioriBuilder(covarianceMatrix.length, threshold, cutoff,
				pruningDifference, contrastEvaluator);

		ChangeChecker changeChecker = new TimeCountChecker(numInstances);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, callback);
		changeChecker.setCallback(streamHiCS);

		int numberSamples = 0;
		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
		}

		System.out.println("Number of elements: " + contrastEvaluator.getNumberOfElements());

		// Evaluation
		return Evaluator.evaluateTPvsFP(streamHiCS.getCurrentlyCorrelatedSubspaces(), correctResult) >= 0.75;
	}
}
