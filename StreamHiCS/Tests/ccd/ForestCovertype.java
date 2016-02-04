package ccd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.Stopwatch;
import environment.AccuracyEvaluator;
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

public class ForestCovertype {

	private String path;
	private ArffFileStream stream;
	private static Stopwatch stopwatch;
	private static final int numberTestRuns = 1;
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
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/RealWorldData/Results.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	/*
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
	*/
	
	public void covertypeNorm() {
		// The change points in the data are: 211840, 495141, 530895, 533642,
		// 543135, 560502
		path = "Tests/RealWorldData/covertypeNorm.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);
		
		StreamSummarisation summarisation = StreamSummarisation.RADIUSCENTROIDS;
		SubspaceBuildup buildup = SubspaceBuildup.APRIORI;
		
		double threshold = 0;
		switch (summarisation) {
		case CLUSTREE_DEPTHFIRST:
			threshold = 0.4;
			break;
		case RADIUSCENTROIDS:
			threshold = 0.1;
			break;
		default:
			break;
		}
		
		int numberOfDimensions = 54;
		int m = 20;
		double alpha = 0.25;
		double epsilon = 0.1;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 4000;
		int checkCount = 10000;
		
		System.out.println("Covertype");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount, summarisation, buildup);		}
	
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
		//cscd.initOption.setValue(0);
		cscd.prepareForUse();
		streamHiCS.setCallback(cscd);

		refDetector = new FullSpaceChangeDetector();
		AbstractClassifier baseLearner = new HoeffdingTree();
		baseLearner.prepareForUse();
		refDetector.baseLearnerOption.setCurrentObject(baseLearner);
		refDetector.prepareForUse();
		
		stopwatch.reset();
		double cscdErrorRate = 0;
		double refErrorRate = 0;
		for (int i = 0; i < numberTestRuns; i++) {
			System.out.println("Run: " +  (i + 1));
			double[] errorRates = testRun();
				cscdErrorRate += errorRates[0];
				refErrorRate += errorRates[1];
		}
		
		cscdErrorRate /= numberTestRuns;
		refErrorRate /= numberTestRuns;
		System.out.println("CSCD: " + cscdErrorRate + ", REF: " + refErrorRate);
	}
	
	private double[] testRun() {
		stream.restart();
		cscd.resetLearning();
		refDetector.resetLearning();

		int numberSamples = 0;

		//ArrayList<Double> cscdDetectedChanges = new ArrayList<Double>();
		//ArrayList<Double> refDetectedChanges = new ArrayList<Double>();
		AccuracyEvaluator cscdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator refAccuracy = new AccuracyEvaluator();
		int numberChangesCSCD = 0;
		int numberChangesREF = 0;

		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance();
			
			
			if(numberSamples % 18000 == 0){
				//System.out.println(cscd.getNumberOfElements());
				//cscd.onAlarm();
			}
			
			// For accuracy
			int trueClass = (int) inst.classValue();
			int prediction = cscd.getClassPrediction(inst);
			cscdAccuracy.addClassLabel(trueClass);
			cscdAccuracy.addPrediction(prediction);
			prediction = Utils.maxIndex(refDetector.getVotesForInstance(inst));
			refAccuracy.addClassLabel(trueClass);
			refAccuracy.addPrediction(prediction);
			
			stopwatch.start("Total_CSCD");
			cscd.trainOnInstance(inst);
			stopwatch.stop("Total_CSCD");
			stopwatch.start("Total_REF");
			refDetector.trainOnInstance(inst);
			stopwatch.stop("Total_REF");

			
			if (cscd.isWarningDetected()) {
				// System.out.println("cscd: WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				numberChangesCSCD++;
				//cscdDetectedChanges.add((double) numberSamples);
				//System.out.println("cscd: CHANGE at " + numberSamples);
			}

			if (refDetector.isWarningDetected()) {
				// System.out.println("refDetector: WARNING at " +
				// numberSamples);
			} else if (refDetector.isChangeDetected()) {
				numberChangesREF++;
				//refDetectedChanges.add((double) numberSamples);
				//System.out.println("refDetector: CHANGE at " + numberSamples);
			}
			 
			numberSamples++;
		}

		System.out.println("CSCD changes: " + numberChangesCSCD);
		System.out.println("REF changes: " + numberChangesREF);
		
		double[] errorRates = new double[2];
		errorRates[0] = cscdAccuracy.calculateOverallErrorRate();
		errorRates[1] = refAccuracy.calculateOverallErrorRate();
		
		List<String> errorRatesList = new ArrayList<String>();
		double[] cscSmoothedErrorRates = cscdAccuracy.calculateSmoothedErrorRates(1000);
		double[] refSmoothedErrorRates = refAccuracy.calculateSmoothedErrorRates(1000);
		for (int i = 0; i < cscdAccuracy.size(); i++) {
			errorRatesList.add(i + "," + cscSmoothedErrorRates[i] + "," + refSmoothedErrorRates[i]);
		}
		
		String filePath = "C:/Users/Vincent/Desktop/ErrorRates.csv";

		try {
			Files.write(Paths.get(filePath), errorRatesList);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		
		return errorRates;
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
			radius = 0.05;
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
}