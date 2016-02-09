package ccd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

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
import moa.core.InstancesHeader;
import moa.streams.ArffFileStream;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.ComponentBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class ElectricityNSW {

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
		// System.out.println(stopwatch.toString());
	}

	@Before
	public void setup() {
		results = new LinkedList<String>();
	}

	@After
	public void after() {
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/RealWorldData/Results.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Test
	public void electricityNWSUnsorted() {
		path = "Tests/RealWorldData/elecNormNew_unsorted.arff";
		// Class index is last attribute but not relevant for this task
		stream = new ArffFileStream(path, -1);

		StreamSummarisation summarisation = StreamSummarisation.CLUSTREE_DEPTHFIRST;
		SubspaceBuildup buildup = SubspaceBuildup.CONNECTED_COMPONENTS;

		double threshold = 0;
		switch (summarisation) {
		case CLUSTREE_DEPTHFIRST:
			threshold = 0.2;
			break;
		case ADAPTINGCENTROIDS:
			threshold = 0.45;
			break;
		default:
			break;
		}

		int numberOfDimensions = 15;
		int m = 50;
		double alpha = 0.1;
		double epsilon = 0.2;
		int cutoff = 8;
		double pruningDifference = 0.15;
		int horizon = 5000;
		int checkCount = 1000;

		System.out.println("Electricity New South Wales unsorted");
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount,
				summarisation, buildup);
	}

	private void carryOutTest(int numberOfDimensions, int m, double alpha, double epsilon, double threshold, int cutoff,
			double pruningDifference, int horizon, int checkCount, StreamSummarisation summarisation,
			SubspaceBuildup buildup) {

		SummarisationAdapter adapter = createSummarisationAdapter(summarisation, numberOfDimensions, horizon);
		Contrast contrastEvaluator = new Contrast(m, alpha, adapter);

		CorrelationSummary correlationSummary = null;
		SubspaceBuilder subspaceBuilder = createSubspaceBuilder(buildup, numberOfDimensions, contrastEvaluator,
				correlationSummary, threshold, cutoff);
		ChangeChecker changeChecker = new TimeCountChecker(checkCount);
		StreamHiCS streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator,
				subspaceBuilder, changeChecker, null, correlationSummary, stopwatch);
		changeChecker.setCallback(streamHiCS);
		cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions, streamHiCS);
		// cscd.initOption.setValue(0);
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
			System.out.println("Run: " + (i + 1));
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

		// ArrayList<Double> cscdDetectedChanges = new ArrayList<Double>();
		// ArrayList<Double> refDetectedChanges = new ArrayList<Double>();
		AccuracyEvaluator cscdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator refAccuracy = new AccuracyEvaluator();
		int numberChangesCSCD = 0;
		int numberChangesREF = 0;

		InstancesHeader header = null;
		Random rand = new Random();
		int totalSize = 15;
		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance();

			if(numberSamples % 1000 == 0){
				System.out.println(cscd.getNumberOfElements());
			}
			
			if (header == null) {
				ArrayList<Attribute> attributes = new ArrayList<Attribute>();
				for (int i = 0; i < 8; i++) {
					attributes.add(inst.dataset().attribute(i));
				}
				for (int i = 8; i < totalSize; i++) {
					Attribute a = new Attribute("noise" + i);
					attributes.add(a);
				}

				// set the class attribute
				attributes.add(inst.dataset().attribute(inst.classIndex()));
				header = new InstancesHeader(new Instances("subspaceData", attributes, 0));
				header.setClassIndex(totalSize);
			}
			double[] newData = new double[totalSize + 1];
			for (int i = 0; i < 8; i++) {
				newData[i] = inst.value(i);
			}
			for (int i = 8; i < totalSize; i++) {
				newData[i] = rand.nextDouble();
			}
			// Setting the class label
			newData[totalSize] = inst.value(inst.classIndex());
			inst = new DenseInstance(inst.weight(), newData);
			inst.setDataset(header);

			if (numberSamples % 18000 == 0) {
				// System.out.println(cscd.getNumberOfElements());
				// cscd.onAlarm();
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
				// cscdDetectedChanges.add((double) numberSamples);
				// System.out.println("cscd: CHANGE at " + numberSamples);
			}

			if (refDetector.isWarningDetected()) {
				// System.out.println("refDetector: WARNING at " +
				// numberSamples);
			} else if (refDetector.isChangeDetected()) {
				numberChangesREF++;
				// refDetectedChanges.add((double) numberSamples);
				// System.out.println("refDetector: CHANGE at " +
				// numberSamples);
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

	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss, int numberOfDimensions,
			int horizon) {
		boolean addDescription = false;
		if (summarisationDescription == null) {
			addDescription = true;
		}
		SummarisationAdapter adapter = null;
		switch (ss) {
		case CLUSTREE_DEPTHFIRST:
			ClusTree clusTree = new ClusTree();
			clusTree.horizonOption.setValue(horizon);
			clusTree.prepareForUse();
			adapter = new MicroclusteringAdapter(clusTree);
			summarisationDescription = "ClusTree, horizon: " + horizon;
			break;
		case ADAPTINGCENTROIDS:
			double radius = 4 * Math.sqrt(numberOfDimensions);
			double learningRate = 1;
			adapter = new CentroidsAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Adapting centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		default:
			adapter = null;
		}
		if (addDescription) {
			results.add(summarisationDescription);
		}
		return adapter;
	}

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb, int numberOfDimensions,
			Contrast contrastEvaluator, CorrelationSummary correlationSummary, double threshold, int cutoff) {
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			builder = new AprioriBuilder(numberOfDimensions, threshold, cutoff, contrastEvaluator, correlationSummary);
			builderDescription = "Apriori, threshold: " + threshold + ", cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, threshold, cutoff, contrastEvaluator,
					correlationSummary, true);
			builderDescription = "Hierarchical, threshold: " + threshold + ", cutoff: " + cutoff;
			break;
		case CONNECTED_COMPONENTS:
			builder = new ComponentBuilder(numberOfDimensions, threshold, contrastEvaluator, correlationSummary);
			builderDescription = "Connnected Components, threshold: " + threshold;
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