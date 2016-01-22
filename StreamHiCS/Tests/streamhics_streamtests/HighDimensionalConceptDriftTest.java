package streamhics_streamtests;

import static org.junit.Assert.*;
import java.util.ArrayList;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import environment.Stopwatch;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.clusterers.clustree.ClusTree;
import moa.streams.ConceptDriftStream;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.GaussianStream;
import streams.UncorrelatedStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class HighDimensionalConceptDriftTest {

	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("onAlarm().");
		}
	};
	private StreamHiCS streamHiCS;
	private ArrayList<SubspaceSet> correctResults;
	private static int testCounter = 0;
	private final int numInstances = 40000;
	private int numberSamples = 0;
	private int numberOfDimensions = 50;
	private static Stopwatch stopwatch;
	private static double tpVSfpSum = 0;
	private static double amjsSum = 0;
	private static double amssSum = 0;
	private static double numElementsSum = 0;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}

	@AfterClass
	public static void calculateAverageScores() {
		System.out.println("Average TPvsFP-score: " + tpVSfpSum / testCounter);
		System.out.println("Average AMJS-score: " + amjsSum / testCounter);
		System.out.println("Average AMSS-score: " + amssSum / testCounter);
		System.out.println("Average #elements: " + numElementsSum / testCounter);
		System.out.println(stopwatch.toString());
	}

	@Before
	public void setUp() throws Exception {
		numberSamples = 0;
	}

	@Test
	public void test1() {
		correctResults = new ArrayList<SubspaceSet>();
		UncorrelatedStream s1 = new UncorrelatedStream();
		s1.dimensionsOption.setValue(50);
		s1.scaleOption.setValue(10);
		s1.prepareForUse();

		double[][] covarianceMatrix2 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, null,
				null, 0.9);
		GaussianStream s2 = new GaussianStream(null, covarianceMatrix2, 1);

		int[] blockBeginnings3 = { 0, 10 };
		int[] blockSizes3 = { 10, 10 };
		double[][] covarianceMatrix3 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions,
				blockBeginnings3, blockSizes3, 0.9);
		GaussianStream s3 = new GaussianStream(null, covarianceMatrix3, 1);

		int[] blockBeginnings4 = { 20, 30 };
		int[] blockSizes4 = { 10, 10 };
		double[][] covarianceMatrix4 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions,
				blockBeginnings4, blockSizes4, 0.9);
		GaussianStream s4 = new GaussianStream(null, covarianceMatrix4, 1);

		ConceptDriftStream cds1 = new ConceptDriftStream();
		cds1.streamOption.setCurrentObject(s1);
		cds1.driftstreamOption.setCurrentObject(s2);
		cds1.positionOption.setValue(10000);
		cds1.widthOption.setValue(500);
		cds1.prepareForUse();

		ConceptDriftStream cds2 = new ConceptDriftStream();
		cds2.streamOption.setCurrentObject(cds1);
		cds2.driftstreamOption.setCurrentObject(s3);
		cds2.positionOption.setValue(20000);
		cds2.widthOption.setValue(500);
		cds2.prepareForUse();

		ConceptDriftStream conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(cds2);
		conceptDriftStream.driftstreamOption.setCurrentObject(s4);
		conceptDriftStream.positionOption.setValue(30000);
		conceptDriftStream.widthOption.setValue(500);
		conceptDriftStream.prepareForUse();

		// Adding the expected results for evaluation
		SubspaceSet cr5000 = new SubspaceSet();
		correctResults.add(cr5000);

		SubspaceSet cr10000 = new SubspaceSet();
		correctResults.add(cr10000);

		SubspaceSet cr15000 = new SubspaceSet();
		correctResults.add(cr15000);

		SubspaceSet cr20000 = new SubspaceSet();
		correctResults.add(cr20000);

		SubspaceSet cr25000 = new SubspaceSet();
		cr25000.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		cr25000.addSubspace(new Subspace(10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
		correctResults.add(cr25000);

		SubspaceSet cr30000 = new SubspaceSet();
		cr30000.addSubspace(new Subspace(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
		cr30000.addSubspace(new Subspace(10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
		correctResults.add(cr30000);

		SubspaceSet cr35000 = new SubspaceSet();
		cr35000.addSubspace(new Subspace(20, 21, 22, 23, 24, 25, 26, 27, 28, 29));
		cr35000.addSubspace(new Subspace(30, 31, 32, 33, 34, 35, 36, 37, 38, 39));
		correctResults.add(cr35000);

		SubspaceSet cr40000 = new SubspaceSet();
		cr40000.addSubspace(new Subspace(20, 21, 22, 23, 24, 25, 26, 27, 28, 29));
		cr40000.addSubspace(new Subspace(30, 31, 32, 33, 34, 35, 36, 37, 38, 39));
		correctResults.add(cr40000);

		int horizon = 6000;
		ClusTree mcs = new ClusTree();
		mcs.horizonOption.setValue(horizon);
		mcs.prepareForUse();

		double alpha = 0.2;
		double epsilon = 0;
		double threshold = 0.2;
		int cutoff = 20;
		double pruningDifference = 0.25;

		SummarisationAdapter adapter = new MicroclusteringAdapter(mcs);
		Contrast contrastEvaluator = new Contrast(50, alpha, adapter);
		ChangeChecker changeChecker = new TimeCountChecker(5000);
		CorrelationSummary correlationSummary = new CorrelationSummary(numberOfDimensions, horizon);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, contrastEvaluator,
				correlationSummary);
		this.streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
				changeChecker, callback, correlationSummary, stopwatch);
		changeChecker.setCallback(streamHiCS);

		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = conceptDriftStream.nextInstance();
			streamHiCS.add(inst);
			numberSamples++;
			if (numberSamples != 0 && numberSamples % 5000 == 0) {
				evaluate();
			}
		}

		fail();
	}

	private void evaluate() {
		System.out.println("Number of samples: " + numberSamples);
		SubspaceSet correctResult = correctResults.get(testCounter);
		SubspaceSet result = streamHiCS.getCurrentlyCorrelatedSubspaces();
		Evaluator.displayResult(result, correctResult);
		double tpVSfp = Evaluator.evaluateTPvsFP(result, correctResult);
		tpVSfpSum += tpVSfp;
		double amjs = Evaluator.evaluateJaccardIndex(result, correctResult);
		amjsSum += amjs;
		double amss = Evaluator.evaluateStructuralSimilarity(result, correctResult);
		amssSum += amss;
		int numElements = streamHiCS.getNumberOfElements();
		numElementsSum += numElements;
		System.out.println("TPvsFP: " + tpVSfp + "; AMJS: " + amjs + "; AMSS: " + amss + "; #Elements: " + numElements);
		testCounter++;
	}
}
