package ccd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.LinkedList;
import java.util.List;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.Stopwatch;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import fullsystem.Contrast;
import fullsystem.CorrelatedSubspacesChangeDetector;
import fullsystem.FullSpaceChangeDetector;
import fullsystem.StreamHiCS;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.clusterers.clustream.Clustream;
import moa.streams.ArffFileStream;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SlidingWindowAdapter;
import streamdatastructures.SummarisationAdapter;
import streamdatastructures.WithDBSCAN;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;
import weka.core.Utils;

public class RealWorldDriftTests {

	private String path;
	private ArffFileStream stream;
	private static Stopwatch stopwatch;
	private static final int numberTestRuns = 10;
	private CorrelatedSubspacesChangeDetector cscd;
	private FullSpaceChangeDetector refDetector;
	private String summarisationDescription = null;
	private String builderDescription = null;
	private List<String> results;
	
	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
	}
	
	@AfterClass
	public static void afterClass() {
		//System.out.println(stopwatch.toString());
	}
	
	@Before
	public void setup(){
		results = new LinkedList<String>();
	}
	
	@After
	public void after(){
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/CCD/RealWorldData/Results.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	@Test
	public void intrusionDetection10PercentFiltered() {
		path = "Tests/RealWorldData/kddcup99_10_percent_filtered.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		StreamSummarisation summarisation = StreamSummarisation.RADIUSCENTROIDS;
		SubspaceBuildup buildup = SubspaceBuildup.APRIORI;
		int numberOfDimensions = 23;
		int m = 50;
		double alpha = 0.15;
		double epsilon = 0.1;
		double threshold = 0.6;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 1000;
		int checkCount = 10000;

		System.out.println("Intrusion Detection 10% filtered");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount, summarisation, buildup);
		System.out.println();
		}
	
	@Test
	public void electricityNWSUnsorted() {
		path = "Tests/RealWorldData/elecNormNew_unsorted.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		StreamSummarisation summarisation = StreamSummarisation.RADIUSCENTROIDS;
		SubspaceBuildup buildup = SubspaceBuildup.APRIORI;
		int numberOfDimensions = 8;
		int m = 50;
		double alpha = 0.15;
		double epsilon = 0.1;
		double threshold = 0.45;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 1000;
		int checkCount = 1000;

		System.out.println("Electricity New South Wales unsorted");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount, summarisation, buildup);
	}
	
	private void carryOutTest(int numberOfDimensions, int m, double alpha, double epsilon, double threshold, int cutoff,
			double pruningDifference, int horizon, int checkCount, StreamSummarisation summarisation, SubspaceBuildup buildup) {

		SummarisationAdapter adapter = createSummarisationAdapter(summarisation, numberOfDimensions, horizon);
		Contrast contrastEvaluator = new Contrast(m, alpha, adapter);
		
		CorrelationSummary correlationSummary = null;
		SubspaceBuilder subspaceBuilder = createSubspaceBuilder(buildup, numberOfDimensions, contrastEvaluator, correlationSummary, threshold, cutoff);
		ChangeChecker changeChecker = new TimeCountChecker(checkCount);
		//ChangeChecker changeChecker = new FullSpaceContrastChecker(checkCount, numberOfDimensions, contrastEvaluator, 0.2, 0.1);
		StreamHiCS streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder, changeChecker, null, correlationSummary, stopwatch);
		changeChecker.setCallback(streamHiCS);
		cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions, streamHiCS);
		cscd.prepareForUse();
		streamHiCS.setCallback(cscd);

		refDetector = new FullSpaceChangeDetector();
		AbstractClassifier baseLearner = new HoeffdingTree();
		baseLearner.prepareForUse();
		refDetector.baseLearnerOption.setCurrentObject(baseLearner);
		refDetector.prepareForUse();
		
		stopwatch.reset();
		double cscdAccuracy = 0;
		double refAccuracy = 0;
		for (int i = 0; i < numberTestRuns; i++) {
			System.out.println("Run: " +  (i + 1));
			double[] accuracies = testRun();
			for (int j = 0; j < 2; j++) {
				cscdAccuracy += accuracies[j];
				refAccuracy += accuracies[j];
			}
		}
		
		cscdAccuracy /= numberTestRuns;
		refAccuracy /= numberTestRuns;
		System.out.println("CSCD: " + cscdAccuracy + ", REF: " + refAccuracy);
	}
	
	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss, int numberOfDimensions, int horizon) {
		boolean addDescription = false;
		if (summarisationDescription == null) {
			addDescription = true;
		}
		SummarisationAdapter adapter = null;
		switch (ss) {
		case SLIDINGWINDOW:
			adapter = new SlidingWindowAdapter(numberOfDimensions, horizon);
			summarisationDescription = "Sliding window, window size: " + horizon;
			break;
		case CLUSTREAM:
			Clustream cluStream = new Clustream();
			cluStream.kernelRadiFactorOption.setValue(2);
			int numberKernels = 400;
			cluStream.maxNumKernelsOption.setValue(numberKernels);
			cluStream.prepareForUse();
			adapter = new MicroclusteringAdapter(cluStream);
			summarisationDescription = "CluStream, maximum number kernels: " + numberKernels;
			break;
		case DENSTREAM:
			WithDBSCAN denStream = new WithDBSCAN();
			int speed = 100;
			double epsilon = 0.5;
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
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case CLUSTREE_BREADTHFIRST:
			clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.breadthFirstSearchOption.set();
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			double radius = 3.5;
			double learningRate = 0.1;
			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		case RADIUSCENTROIDS:
			radius = 3.5;
			adapter = new CentroidsAdapter(horizon, radius, 0.1, "radius");
			summarisationDescription = "Radius centroids, horizon: " + horizon + ", radius: " + radius;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(summarisationDescription);
		}
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb, int numberOfDimensions, Contrast contrastEvaluator, CorrelationSummary correlationSummary, double threshold, int cutoff) {
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			builder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, contrastEvaluator,
					correlationSummary);
			builderDescription = "Apriori, threshold: " + threshold + ", cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, threshold, cutoff,
					contrastEvaluator, correlationSummary, true);
			builderDescription = "Hierarchical, threshold: " + threshold + ", cutoff: " + cutoff;
			break;
		default:
			builder = null;
		}
		if (addDescription) {
			results.add(builderDescription);
		}
		return builder;
	}

	private double[] testRun() {
		stream.restart();
		cscd.resetLearning();
		refDetector.resetLearning();

		int numberSamples = 0;

		//ArrayList<Double> cscdDetectedChanges = new ArrayList<Double>();
		//ArrayList<Double> refDetectedChanges = new ArrayList<Double>();
		int numberCorrectCSCD = 0;
		int numberCorrectREF = 0;
		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance();
			
			// For accuracy
			int trueClass = (int) inst.classValue();
			if(trueClass == cscd.getClassPrediction(inst)){
				numberCorrectCSCD++;
			}
			if(trueClass == Utils.maxIndex(refDetector.getVotesForInstance(inst))){
				numberCorrectREF++;
			}
			
			stopwatch.start("Total_CSCD");
			cscd.trainOnInstance(inst);
			stopwatch.stop("Total_CSCD");
			stopwatch.start("Total_REF");
			refDetector.trainOnInstance(inst);
			stopwatch.stop("Total_REF");

			/*
			if (cscd.isWarningDetected()) {
				// System.out.println("cscd: WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				cscdDetectedChanges.add((double) numberSamples);
				System.out.println("cscd: CHANGE at " + numberSamples);
			}

			if (refDetector.isWarningDetected()) {
				// System.out.println("refDetector: WARNING at " +
				// numberSamples);
			} else if (refDetector.isChangeDetected()) {
				refDetectedChanges.add((double) numberSamples);
				System.out.println("refDetector: CHANGE at " + numberSamples);
			}
			 */
			numberSamples++;
		}

		double[] accuracies = new double[2];
		accuracies[0] = ((double) numberCorrectCSCD) / numberSamples;
		accuracies[1] = ((double) numberCorrectREF) / numberSamples;
		return accuracies;
	}
}
