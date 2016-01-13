package streamtests;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.Evaluator;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.SubspaceRandomRBFGeneratorDrift;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class SubspaceRBF {

	private SubspaceRandomRBFGeneratorDrift stream;
	private StreamHiCS streamHiCS;
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
	private Random random = new Random();
	private static Stopwatch stopwatch;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}

	@AfterClass
	public static void afterClass() {
		System.out.println(stopwatch.toString());
	}

	@Before
	public void setUp() throws Exception {
		alpha = 0.15;
		epsilon = 0;
		threshold = 0.2;
		cutoff = 8;
		pruningDifference = 0.15;

		int numberOfDimensions = 10;
		stream = new SubspaceRandomRBFGeneratorDrift();
		stream.numAttsOption.setValue(numberOfDimensions);
		stream.avgSubspaceSizeOption.setValue(5);
		stream.numCentroidsOption.setValue(10);
		stream.numSubspaceCentroidsOption.setValue(5);
		stream.sameSubspaceOption.set();
		stream.prepareForUse();

		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(2000);
		mcs.resetLearning();
		SummarisationAdapter adapter = new MicroclusteringAdapter(mcs);

		Contrast contrastEvaluator = new Contrast(m, alpha, adapter);

		CorrelationSummary correlationSummary = new CorrelationSummary(numberOfDimensions);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, contrastEvaluator,
				correlationSummary);
		ChangeChecker changeChecker = new TimeCountChecker(numInstances);
		streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, callback, correlationSummary, stopwatch);
		changeChecker.setCallback(streamHiCS);
	}

	@Test
	public void test() {
		int numberTestRuns = 1;
		double sumTPvsFP = 0;
		double sumAMJS = 0;
		double sumAMSS = 0;

		for (int i = 0; i < numberTestRuns; i++) {
			double[] performanceMeasures = testRun();
			sumTPvsFP += performanceMeasures[0];
			sumAMJS += performanceMeasures[1];
			sumAMSS += performanceMeasures[2];
		}

		double sumEvaluationTime = stopwatch.getTime("Evaluation");
		double sumAddingTime = stopwatch.getTime("Adding");
		double sumTotalTime = stopwatch.getTime("Total");

		System.out.println("Average TPvsFP: " + sumTPvsFP / numberTestRuns);
		System.out.println("Average AMJS: " + sumAMJS / numberTestRuns);
		System.out.println("Average AMSS: " + sumAMSS / numberTestRuns);
		System.out.println("Average evaluation time: " + sumEvaluationTime / numberTestRuns);
		System.out.println("Average adding time: " + sumAddingTime / numberTestRuns);
		System.out.println("Average total time: " + sumTotalTime / numberTestRuns);
		
		fail("No condition");
	}

	private double[] testRun() {
		stream.instanceRandomSeedOption.setValue(random.nextInt(1000000000));
		stream.prepareForUse();
		streamHiCS.clear();
		
		int numberSamples = 0;

		while (stream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = stream.nextInstance();
			stopwatch.start("Total");
			streamHiCS.add(inst);
			stopwatch.stop("Total");
			numberSamples++;
		}
		
		System.out.println("Number of elements: " + streamHiCS.getNumberOfElements());
		
		SubspaceSet result = streamHiCS.getCurrentlyCorrelatedSubspaces();
		SubspaceSet correctResult = stream.getSubspaces();
		for(Subspace s : correctResult.getSubspaces()){
			s.sort();
		}
		correctResult.sort();
		double[] performanceMeasures = new double[3];
		Evaluator.displayResult(result, correctResult);
		performanceMeasures[0] = Evaluator.evaluateTPvsFP(result, correctResult);
		performanceMeasures[1] = Evaluator.evaluateJaccardIndex(result, correctResult);
		performanceMeasures[2] = Evaluator.evaluateStructuralSimilarity(result, correctResult);
		
		return performanceMeasures;
	}

}
