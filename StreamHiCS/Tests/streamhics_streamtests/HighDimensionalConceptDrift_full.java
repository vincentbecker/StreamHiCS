package streamhics_streamtests;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import environment.Stopwatch;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.clusterers.clustream.Clustream;
import clustree.ClusTree;
import moa.streams.ConceptDriftStream;
import streamdatastructures.MCAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streamdatastructures.WithDBSCAN;
import streams.GaussianStream;
import streams.UncorrelatedStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class HighDimensionalConceptDrift_full {

	private StreamHiCS streamHiCS;
	private int numberOfDimensions = 50;
	private static Stopwatch stopwatch;
	private CorrelationSummary correlationSummary;
	private final int horizon = 6000;
	private final int m = 50;
	private final double alpha = 0.1;
	private double epsilon = 0;
	private double aprioriThreshold;
	private double hierarchicalThreshold;
	private int cutoff;
	private double pruningDifference;
	private Contrast contrastEvaluator;
	private static final int numberTestRuns = 10;
	private List<String> results;
	private String summarisationDescription = null;
	private String builderDescription = null;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("onAlarm().");
		}
	};
	private ArrayList<SubspaceSet> correctResults;
	private int testCounter = 0;
	private final int numInstances = 40000;
	private int numberSamples = 0;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}

	@AfterClass
	public static void calculateAverageScores() {
		// System.out.println("Average TPvsFP-score: " + tpVSfpSum /
		// testCounter);
		// System.out.println("Average AMJS-score: " + amjsSum / testCounter);
		// System.out.println(stopwatch.toString());
	}

	@Before
	public void setUp() throws Exception {
		numberSamples = 0;
	}

	@Test
	public void test() {
		// Output
		results = new LinkedList<String>();
		SummarisationAdapter adapter;
		SubspaceBuilder subspaceBuilder;
		for (StreamSummarisation summarisation : StreamSummarisation.values()) {
			summarisationDescription = null;
			for (SubspaceBuildup buildup : SubspaceBuildup.values()) {
				builderDescription = null;
				if (summarisation == StreamSummarisation.RADIUSCENTROIDS && buildup == SubspaceBuildup.APRIORI) {
				stopwatch.reset();
				double sumTPvsFP = 0;
				double sumAMJS = 0;
				double sumAMSS = 0;
				double sumElements = 0;

				double threshold = 1;
				switch (buildup) {
				case APRIORI:
					threshold = aprioriThreshold;
					break;
				case HIERARCHICAL:
					threshold = hierarchicalThreshold;
					break;
				}

				// Creating the StreamHiCS system
				adapter = createSummarisationAdapter(summarisation);
				contrastEvaluator = new Contrast(m, alpha, adapter);
				correlationSummary = new CorrelationSummary(numberOfDimensions, horizon);
				subspaceBuilder = createSubspaceBuilder(buildup);
				ChangeChecker changeChecker = new TimeCountChecker(5000);
				streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder,
						changeChecker, callback, correlationSummary, stopwatch);
				changeChecker.setCallback(streamHiCS);

				for (int i = 0; i < numberTestRuns; i++) {
					double[] performanceMeasures = testRun();
					sumTPvsFP += performanceMeasures[0];
					sumAMJS += performanceMeasures[1];
					sumAMSS += performanceMeasures[2];
					sumElements += performanceMeasures[3];
				}

				// Calculate results
				double sumEvaluationTime = stopwatch.getTime("Evaluation");
				double sumAddingTime = stopwatch.getTime("Adding");
				double sumTotalTime = stopwatch.getTime("Total");

				double avgTPvsFP = sumTPvsFP / numberTestRuns;
				double avgAMJS = sumAMJS / numberTestRuns;
				double avgAMSS = sumAMSS / numberTestRuns;
				double avgNumElements = sumElements / numberTestRuns;
				double avgEvalTime = sumEvaluationTime / numberTestRuns;
				double avgAddingTime = sumAddingTime / numberTestRuns;
				double avgTotalTime = sumTotalTime / numberTestRuns;

				String measures = avgTPvsFP + ", " + avgAMJS + ", " + avgAMSS + ", " + avgNumElements + ", "
						+ avgEvalTime + ", " + avgAddingTime + ", " + avgTotalTime;
				System.out.println(measures);
				results.add(measures);
				 }
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/StreamHiCS/GaussianStreams/HighDimensionalDrift/Results.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private double[] testRun() {
		streamHiCS.clear();
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

		numberSamples = 0;
		testCounter = 0;
		double[] performanceMeasures;
		double[] performanceMeasureSums = new double[4];
		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = conceptDriftStream.nextInstance();
			stopwatch.start("Total");
			streamHiCS.add(inst);
			stopwatch.stop("Total");
			numberSamples++;
			if (numberSamples != 0 && numberSamples % 5000 == 0) {
				performanceMeasures = evaluate();
				for (int i = 0; i < performanceMeasures.length; i++) {
					performanceMeasureSums[i] += performanceMeasures[i];
				}
				testCounter++;
			}
		}

		for (int i = 0; i < performanceMeasureSums.length; i++) {
			performanceMeasureSums[i] /= 8;
		}

		return performanceMeasureSums;
	}

	private double[] evaluate() {
		// System.out.println("Number of samples: " + numberSamples);
		SubspaceSet result = streamHiCS.getCurrentlyCorrelatedSubspaces();
		SubspaceSet correctResult = correctResults.get(testCounter);
		for (Subspace s : correctResult.getSubspaces()) {
			s.sort();
		}
		correctResult.sort();
		//Evaluator.displayResult(result, correctResult);
		double[] performanceMeasures = new double[4];
		performanceMeasures[0] = Evaluator.evaluateTPvsFP(result, correctResult);
		performanceMeasures[1] = Evaluator.evaluateJaccardIndex(result, correctResult);
		performanceMeasures[2] = Evaluator.evaluateStructuralSimilarity(result, correctResult);
		performanceMeasures[3] = streamHiCS.getNumberOfElements();

		return performanceMeasures;
	}

	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss) {
		boolean addDescription = false;
		if (summarisationDescription == null) {
			addDescription = true;
		}
		SummarisationAdapter adapter = null;
		switch (ss) {
		case SLIDINGWINDOW:
			aprioriThreshold = 0.2;
			hierarchicalThreshold = 0.25;
			adapter = new SlidingWindowAdapter(numberOfDimensions, horizon);
			summarisationDescription = "Sliding window, window size: " + horizon;
			break;
		case CLUSTREAM:
			aprioriThreshold = 0.2;
			hierarchicalThreshold = 0.25;
			Clustream cluStream = new Clustream();
			cluStream.kernelRadiFactorOption.setValue(2);
			int numberKernels = 400;
			cluStream.maxNumKernelsOption.setValue(numberKernels);
			cluStream.prepareForUse();
			adapter = new MicroclusteringAdapter(cluStream);
			summarisationDescription = "CluStream, maximum number kernels: " + numberKernels;
			break;
		case DENSTREAM:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.5;
			WithDBSCAN denStream = new WithDBSCAN();
			int speed = 100;
			double epsilon = 2;
			double beta = 0.2;
			double mu = 10;
			denStream.speedOption.setValue(speed);
			denStream.epsilonOption.setValue(epsilon);
			denStream.betaOption.setValue(beta);
			denStream.muOption.setValue(mu);
			// lambda calculated from horizon
			double lambda = -Math.log(0.01) / Math.log(2) / (double) horizon;
			denStream.lambdaOption.setValue(lambda);
			denStream.prepareForUse();
			adapter = new MicroclusteringAdapter(denStream);
			summarisationDescription = "DenStream, speed: " + speed + ", epsilon: " + epsilon + ", beta" + beta + ", mu"
					+ mu + ", lambda" + lambda;
			break;
		case CLUSTREE_DEPTHFIRST:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.35;
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree depthFirst, horizon: " + horizon;
			break;
		case CLUSTREE_BREADTHFIRST:
			aprioriThreshold = 0.2;
			hierarchicalThreshold = 0.25;
			clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.breadthFirstSearchOption.set();
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree breadthFirst, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			aprioriThreshold = 0.3;
			hierarchicalThreshold = 0.35;
			double radius = 45;
			double learningRate = 1;
			adapter = new MCAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Adapting centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		case RADIUSCENTROIDS:
			aprioriThreshold = 0.25;
			hierarchicalThreshold = 0.3;
			radius = 5.5;
			adapter = new MCAdapter(horizon, radius, 0.1, "radius");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(summarisationDescription);
			addDescription = false;
		}
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb) {
		cutoff = 20;
		pruningDifference = 0.25;
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			builder = new AprioriBuilder(numberOfDimensions, aprioriThreshold, cutoff, contrastEvaluator,
					correlationSummary);
			builderDescription = "Apriori, threshold:" + aprioriThreshold + "cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, hierarchicalThreshold, cutoff,
					contrastEvaluator, correlationSummary, true);
			builderDescription = "Hierarchical, threshold: " + hierarchicalThreshold + ", cutoff: " + cutoff;
			break;
		default:
			builder = null;
		}
		if (addDescription) {
			results.add(builderDescription);
			addDescription = false;
		}
		return builder;
	}
}
