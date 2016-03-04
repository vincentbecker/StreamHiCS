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
import changedetection.FullSpaceChangeDetector;
import changedetection.SubspaceChangeDetectors;
import changedetection.SubspaceClassifiersChangeDetector;
import clustree.ClusTree;
import environment.Stopwatch;
import environment.AccuracyEvaluator;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import fullsystem.Contrast;
import fullsystem.StreamHiCS;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.InstancesHeader;
import moa.streams.ArffFileStream;
import streamdatastructures.MCAdapter;
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

public class RealWorldDatasets {

	private enum Dataset {
		COVERTYPE, INTRUSIONDETECTION, ELECTRICITY
	};

	private String path;
	private ArffFileStream stream;
	private static Stopwatch stopwatch;
	private static final int numberTestRuns = 10;
	private SubspaceChangeDetectors scd;
	private SubspaceChangeDetectors scdr;
	private SubspaceClassifiersChangeDetector sccd;
	private SubspaceClassifiersChangeDetector sccdr;
	private FullSpaceChangeDetector refDetector;
	private HoeffdingTree refClassifier;
	private String summarisationDescription = null;
	private String builderDescription = null;
	private List<String> results;
	private int numberOfDimensions;
	private int originalNumDimensions;
	private Dataset dataset;
	private String outputPath;

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
		try {
			Files.write(Paths.get(outputPath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Test
	public void runDataset() {
		dataset = Dataset.INTRUSIONDETECTION;
		StreamSummarisation summarisation = StreamSummarisation.ADAPTINGCENTROIDS;
		SubspaceBuildup buildup = SubspaceBuildup.CONNECTED_COMPONENTS;
		double threshold = 0;
		int m = 0;
		double alpha = 0;
		double epsilon = 0;
		int cutoff = 0;
		double pruningDifference = 0;
		int horizon = 0;
		int checkCount = 0;
		double radius = 0;

		switch (dataset) {
		case COVERTYPE:
			//path = "Tests/RealWorldData/covertypeNorm_filtered.arff";
			path = "Tests/RealWorldData/covertypeNorm.arff";
			// Class index is last attribute but not relevant for this task
			stream = new ArffFileStream(path, -1);

			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/RealWorldData/ForestCovertype_Results.txt";
			
			summarisation = StreamSummarisation.CLUSTREE_DEPTHFIRST;
			buildup = SubspaceBuildup.CONNECTED_COMPONENTS;

			//numberOfDimensions = 10;
			numberOfDimensions = 54;
			//originalNumDimensions = 10;
			originalNumDimensions = 54;
			switch (summarisation) {
			case CLUSTREE_DEPTHFIRST:
				if(numberOfDimensions == 10){
					threshold = 0.3;
				}else{
					threshold = 0.65;
				}
				break;
			case ADAPTINGCENTROIDS:
				if (numberOfDimensions == 10) {
					threshold = 0.3;
					//radius = 14 * Math.sqrt(numberOfDimensions);
					radius = 0.2;
				} else {
					threshold = 0.25;
					radius = 14 * Math.sqrt(numberOfDimensions);
				}
				break;
			case RADIUSCENTROIDS:
				if(numberOfDimensions == originalNumDimensions){
					threshold = 0.35;
					radius = 0.25;
				}else{
					threshold = 0.35;
					radius = 0.1;
				}

				break;
			default:
				break;
			}

			m = 50;
			horizon = 10000;
			checkCount = 10000;
			if (numberOfDimensions == 10) {
				alpha = 0.1;
				epsilon = 0.2;
				cutoff = 8;
				pruningDifference = 0.15;
			} else {
				alpha = 0.1;
				epsilon = 0.2;
				cutoff = 8;
				pruningDifference = 0.15;
			}

			System.out.println("ForestCovertype filtered unsorted");
			break;
		case INTRUSIONDETECTION:
			path = "Tests/RealWorldData/kddcup99_10_percent.arff";
			// Class index is last attribute but not relevant for this task
			stream = new ArffFileStream(path, -1);

			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/RealWorldData/IntrusionDetection_Results.txt";
			
			summarisation = StreamSummarisation.RADIUSCENTROIDS;
			buildup = SubspaceBuildup.CONNECTED_COMPONENTS;

			//numberOfDimensions = 23;
			//originalNumDimensions = 23;
			numberOfDimensions = 34;
			originalNumDimensions = 34;
			switch (summarisation) {
			case CLUSTREE_DEPTHFIRST:
				threshold = 0.55;
				break;
			case ADAPTINGCENTROIDS:
				if (numberOfDimensions == 23) {
					threshold = 0.6;
					radius = 180;
				} else {
					threshold = 0.6;
					radius = 1000 * Math.sqrt(numberOfDimensions) - 1;
				}
				break;
			case RADIUSCENTROIDS:
				if (numberOfDimensions == 23) {
					threshold = 0.6;
					radius = 100;
				} else {
					threshold = 0.6;
					radius = 100;
				}
				break;
			default:
				break;
			}
			m = 50;
			alpha = 0.1;
			epsilon = 0.2;
			cutoff = 8;
			pruningDifference = 0.15;
			horizon = 10000;
			checkCount = 10000;

			System.out.println("Intrusion Detection filtered unsorted");

			break;
		case ELECTRICITY:
			path = "Tests/RealWorldData/elecNormNew_unsorted.arff";
			// Class index is last attribute but not relevant for this task
			stream = new ArffFileStream(path, -1);

			outputPath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/RealWorldData/Electricity_Results.txt";
			
			summarisation = StreamSummarisation.RADIUSCENTROIDS;
			buildup = SubspaceBuildup.CONNECTED_COMPONENTS;

			numberOfDimensions = 8;
			originalNumDimensions = 8;
			switch (summarisation) {
			case CLUSTREE_DEPTHFIRST:
				threshold = 0.5;
				break;
			case ADAPTINGCENTROIDS:
				if (numberOfDimensions == 8) {
					threshold = 0.4;
					//radius = 3 * Math.sqrt(numberOfDimensions) - 1;
					radius = 0.1;
				} else {
					threshold = 0.5;
					radius = 2;
				}
				// threshold = 0.45 + alpha = 0.15 + apriori-> best for
				// eight dimensions
				break;
			case RADIUSCENTROIDS:
				if (numberOfDimensions == 8) {
					threshold = 0.4;
					radius = 0.1;
				} else if(numberOfDimensions == 20) {
					threshold = 0.4;
					radius = 0.6;
				}else{
					threshold = 0.4;
					radius = 0.55;
				}
				// threshold = 0.45 + alpha = 0.15 + apriori-> best for
				// eight dimensions
				break;
			default:
				break;
			}

			// 50 dims
			// threshold = 0.4, alpha = 0.15

			m = 50;
			alpha = 0.15;
			epsilon = 0.25;
			cutoff = 8;
			pruningDifference = 0.15;
			horizon = 2000;
			checkCount = 1000;

			System.out.println("Electricity New South Wales unsorted");
			break;
		default:
			break;

		}
		carryOutTest(numberOfDimensions, m, alpha, epsilon, threshold, cutoff, pruningDifference, horizon, checkCount,
				summarisation, radius, buildup);
	}

	private void carryOutTest(int numberOfDimensions, int m, double alpha, double epsilon, double threshold, int cutoff,
			double pruningDifference, int horizon, int checkCount, StreamSummarisation summarisation, double radius,
			SubspaceBuildup buildup) {

		// Creating the SCD system
		SummarisationAdapter adapter1 = createSummarisationAdapter(summarisation, numberOfDimensions, horizon, radius);
		Contrast contrastEvaluator1 = new Contrast(m, alpha, adapter1);
		// CorrelationSummary correlationSummary1 = new
		// CorrelationSummary(numberOfDimensions, horizon);
		CorrelationSummary correlationSummary1 = null;
		SubspaceBuilder subspaceBuilder1 = createSubspaceBuilder(buildup, numberOfDimensions, contrastEvaluator1,
				correlationSummary1, threshold, cutoff);
		ChangeChecker changeChecker1 = new TimeCountChecker(horizon);
		StreamHiCS streamHiCS1 = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator1,
				subspaceBuilder1, changeChecker1, null, correlationSummary1, stopwatch);
		changeChecker1.setCallback(streamHiCS1);
		scd = new SubspaceChangeDetectors(numberOfDimensions, streamHiCS1);
		scd.useRestspaceOption.setValue(false);
		scd.initOption.setValue(5000);
		scd.addOption.setValue(true);
		scd.prepareForUse();
		streamHiCS1.addCallback(scd);

		// SCDr with restspace
		scdr = new SubspaceChangeDetectors(numberOfDimensions, streamHiCS1);
		scdr.useRestspaceOption.setValue(true);
		scdr.initOption.setValue(5000);
		scd.addOption.setValue(false);
		scdr.prepareForUse();
		streamHiCS1.addCallback(scdr);

		// Creating the SCCD system
		sccd = new SubspaceClassifiersChangeDetector(numberOfDimensions, streamHiCS1);
		sccd.useRestspaceOption.setValue(false);
		sccd.initOption.setValue(5000);
		sccd.prepareForUse();
		streamHiCS1.addCallback(sccd);

		// SCCDr
		sccdr = new SubspaceClassifiersChangeDetector(numberOfDimensions, streamHiCS1);
		sccdr.useRestspaceOption.setValue(true);
		sccdr.initOption.setValue(5000);
		sccdr.prepareForUse();
		streamHiCS1.addCallback(sccdr);

		// Creating the reference detector
		refDetector = new FullSpaceChangeDetector();
		AbstractClassifier baseLearner = new HoeffdingTree();
		baseLearner.prepareForUse();
		refDetector.baseLearnerOption.setCurrentObject(baseLearner);
		refDetector.prepareForUse();

		refClassifier = new HoeffdingTree();
		refClassifier.prepareForUse();

		stopwatch.reset();
		double[][] totalMeasures = new double[6][2];
		for (int i = 0; i < numberTestRuns; i++) {
			System.out.println("Run: " + (i + 1));
			double[][] measures = testRun();
			for (int j = 0; j < 6; j++) {
				for (int k = 0; k < 2; k++) {
					totalMeasures[j][k] += measures[j][k];
				}
			}
		}

		for (int j = 0; j < 6; j++) {
			for (int k = 0; k < 2; k++) {
				totalMeasures[j][k] /= numberTestRuns;
			}
		}

		double evaluationTime = stopwatch.getTime("Evaluation") / numberTestRuns;
		double addingTime = stopwatch.getTime("Adding") / numberTestRuns;
		double scdRuntime = stopwatch.getTime("Total_SCD") / numberTestRuns;
		double scdrRuntime = stopwatch.getTime("Total_SCDr") / numberTestRuns;
		double sccdRuntime = stopwatch.getTime("Total_SCCD") / numberTestRuns;
		double sccdrRuntime = stopwatch.getTime("Total_SCCDr") / numberTestRuns;
		double refRuntime = stopwatch.getTime("Total_REF") / numberTestRuns;
		double refClassifierRuntime = stopwatch.getTime("Total_REFCLASSIFIER") / numberTestRuns;

		System.out.println("Number changes, error rate, Evaluation time, Adding time, Total time");
		String scdMeasures = "SCD, " + totalMeasures[0][0] + ", " + totalMeasures[0][1] + ", " + evaluationTime + ", "
				+ addingTime + ", " + scdRuntime;
		String scdrMeasures = "SCDr, " + totalMeasures[1][0] + ", " + totalMeasures[1][1] + ", " + evaluationTime + ", "
				+ addingTime + ", " + (scdrRuntime + evaluationTime + addingTime);
		String sccdMeasures = "SCCD, " + totalMeasures[2][0] + ", " + totalMeasures[2][1] + ", " + evaluationTime + ", "
				+ addingTime + ", " + (sccdRuntime + evaluationTime + addingTime);
		String sccdrMeasures = "SCCDr, " + totalMeasures[3][0] + ", " + totalMeasures[3][1] + ", " + evaluationTime
				+ ", " + addingTime + ", " + (sccdrRuntime + evaluationTime + addingTime);
		String refMeasures = "REF, " + totalMeasures[4][0] + ", " + totalMeasures[4][1] + ", " + 0 + ", " + 0 + ", "
				+ refRuntime;
		String refClassifierMeasures = "REF classifier, " + 0 + ", " + totalMeasures[5][1] + ", " + 0 + ", " + 0 + ", "
				+ refClassifierRuntime;
		System.out.println(scdMeasures);
		System.out.println(scdrMeasures);
		System.out.println(sccdMeasures);
		System.out.println(sccdrMeasures);
		System.out.println(refMeasures);
		System.out.println(refClassifierMeasures);
		results.add(scdMeasures);
		results.add(scdrMeasures);
		results.add(sccdMeasures);
		results.add(sccdrMeasures);
		results.add(refMeasures);
		results.add(refClassifierMeasures);
	}

	private double[][] testRun() {
		stream.restart();
		scd.resetLearning();
		scdr.resetLearning();
		sccd.resetLearning();
		sccdr.resetLearning();
		refDetector.resetLearning();

		int numberSamples = 0;

		AccuracyEvaluator scdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator scdrAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator sccdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator sccdrAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator refAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator refClassifierAccuracy = new AccuracyEvaluator();
		int numberChangesSCD = 0;
		int numberChangesSCDr = 0;
		int numberChangesSCCD = 0;
		int numberChangesSCCDr = 0;
		int numberChangesREF = 0;

		InstancesHeader header = null;
		Random rand = new Random();
		while (stream.hasMoreInstances()) {
			Instance inst = stream.nextInstance();
			
			if (numberSamples % 10000 == 0) {
				System.out.println(scd.getNumberOfElements());
			}

			if (numberOfDimensions > originalNumDimensions) {
				if (header == null) {
					ArrayList<Attribute> attributes = new ArrayList<Attribute>();
					for (int i = 0; i < originalNumDimensions; i++) {
						Attribute a = new Attribute("original" + i);
						attributes.add(a);
					}
					for (int i = originalNumDimensions; i < numberOfDimensions; i++) {
						Attribute a = new Attribute("noise" + i);
						attributes.add(a);
					}
					ArrayList<String> classLabels = new ArrayList<String>();
					switch (dataset) {
					case COVERTYPE:
						classLabels.add("1");
						classLabels.add("2");
						classLabels.add("3");
						classLabels.add("4");
						classLabels.add("5");
						classLabels.add("6");
						classLabels.add("7");
						break;
					case INTRUSIONDETECTION:
						classLabels.add("back");
						classLabels.add("buffer_overflow");
						classLabels.add("ftp_write");
						classLabels.add("guess_passwd");
						classLabels.add("imap");
						classLabels.add("ipsweep");
						classLabels.add("land");
						classLabels.add("loadmodule");
						classLabels.add("multihop");
						classLabels.add("neptune");
						classLabels.add("nmap");
						classLabels.add("normal");
						classLabels.add("perl");
						classLabels.add("phf");
						classLabels.add("pod");
						classLabels.add("portsweep");
						classLabels.add("rootkit");
						classLabels.add("satan");
						classLabels.add("smurf");
						classLabels.add("spy");
						classLabels.add("teardrop");
						classLabels.add("warezclient");
						classLabels.add("warezmaster");
						break;
					case ELECTRICITY:
						classLabels.add("UP");
						classLabels.add("DOWN");
						break;
					default:
						break;

					}
					attributes.add(new Attribute("class", classLabels));
					header = new InstancesHeader(new Instances("AugmentedStream", attributes, 0));
					header.setClassIndex(numberOfDimensions);
				}
				double[] newData = new double[numberOfDimensions + 1];
				for (int i = 0; i < originalNumDimensions; i++) {
					newData[i] = inst.value(i);
				}
				for (int i = originalNumDimensions; i < numberOfDimensions; i++) {
					newData[i] = rand.nextDouble();
				}
				// Setting the class label
				newData[numberOfDimensions] = inst.value(inst.classIndex());
				inst = new DenseInstance(inst.weight(), newData);
				inst.setDataset(header);
			}

			// For accuracy
			int trueClass = (int) inst.classValue();
			int prediction = scd.getClassPrediction(inst);
			scdAccuracy.addClassLabel(trueClass);
			scdAccuracy.addPrediction(prediction);
			prediction = scdr.getClassPrediction(inst);
			scdrAccuracy.addClassLabel(trueClass);
			scdrAccuracy.addPrediction(prediction);
			prediction = sccd.getClassPrediction(inst);
			sccdAccuracy.addClassLabel(trueClass);
			sccdAccuracy.addPrediction(prediction);
			prediction = sccdr.getClassPrediction(inst);
			sccdrAccuracy.addClassLabel(trueClass);
			sccdrAccuracy.addPrediction(prediction);
			prediction = Utils.maxIndex(refDetector.getVotesForInstance(inst));
			refAccuracy.addClassLabel(trueClass);
			refAccuracy.addPrediction(prediction);
			prediction = Utils.maxIndex(refClassifier.getVotesForInstance(inst));
			refClassifierAccuracy.addClassLabel(trueClass);
			refClassifierAccuracy.addPrediction(prediction);

			stopwatch.start("Total_SCD");
			scd.trainOnInstance(inst);
			stopwatch.stop("Total_SCD");
			stopwatch.start("Total_SCDr");
			scdr.trainOnInstance(inst);
			stopwatch.stop("Total_SCDr");
			stopwatch.start("Total_SCCD");
			sccd.trainOnInstance(inst);
			stopwatch.stop("Total_SCCD");
			stopwatch.start("Total_SCCDr");
			sccdr.trainOnInstance(inst);
			stopwatch.stop("Total_SCCDr");
			stopwatch.start("Total_REF");
			refDetector.trainOnInstance(inst);
			stopwatch.stop("Total_REF");
			stopwatch.start("Total_REFCLASSIFIER");
			refClassifier.trainOnInstance(inst);
			stopwatch.stop("Total_REFCLASSIFIER");

			if (scd.isChangeDetected()) {
				numberChangesSCD++;
			}
			if (scdr.isChangeDetected()) {
				numberChangesSCDr++;
			}
			if (sccd.isChangeDetected()) {
				numberChangesSCCD++;
			}
			if (sccdr.isChangeDetected()) {
				numberChangesSCCDr++;
			}
			if (refDetector.isChangeDetected()) {
				numberChangesREF++;
			}

			numberSamples++;
		}

		double[][] measures = new double[6][2];
		measures[0][0] = numberChangesSCD;
		measures[1][0] = numberChangesSCDr;
		measures[2][0] = numberChangesSCCD;
		measures[3][0] = numberChangesSCCDr;
		measures[4][0] = numberChangesREF;

		measures[0][1] = scdAccuracy.calculateOverallErrorRate();
		measures[1][1] = scdrAccuracy.calculateOverallErrorRate();
		measures[2][1] = sccdAccuracy.calculateOverallErrorRate();
		measures[3][1] = sccdrAccuracy.calculateOverallErrorRate();
		measures[4][1] = refAccuracy.calculateOverallErrorRate();
		measures[5][1] = refClassifierAccuracy.calculateOverallErrorRate();

		String scdP = "SCD, " + measures[0][0] + ", " + measures[0][1];
		String scdrP = "SCDr, " + measures[1][0] + ", " + measures[1][1];
		String sccdP = "SCCD, " + measures[2][0] + ", " + measures[2][1];
		String sccdrP = "SCCDr, " + measures[3][0] + ", " + measures[3][1];
		String refP = "REF, " + +measures[4][0] + ", " + measures[4][1];
		String refClassifierP = "REF classifier, " + measures[5][0] + ", " + measures[5][1];
		System.out.println(scdP);
		System.out.println(scdrP);
		System.out.println(sccdP);
		System.out.println(sccdrP);
		System.out.println(refP);
		System.out.println(refClassifierP);
		results.add(scdP);
		results.add(scdrP);
		results.add(sccdP);
		results.add(sccdrP);
		results.add(refP);
		results.add(refClassifierP);

		List<String> errorRatesList = new ArrayList<String>();
		double[] scdSmoothedErrorRates = scdAccuracy.calculateSmoothedErrorRates(1000);
		double[] scdrSmoothedErrorRates = scdAccuracy.calculateSmoothedErrorRates(1000);
		double[] sccdSmoothedErrorRates = sccdAccuracy.calculateSmoothedErrorRates(1000);
		double[] sccdrSmoothedErrorRates = sccdAccuracy.calculateSmoothedErrorRates(1000);
		double[] refSmoothedErrorRates = refAccuracy.calculateSmoothedErrorRates(1000);
		double[] refClassifierSmoothedErrorRates = refClassifierAccuracy.calculateSmoothedErrorRates(1000);
		for (int i = 0; i < scdAccuracy.size(); i++) {
			errorRatesList.add(i + "," + scdSmoothedErrorRates[i] + "," + scdrSmoothedErrorRates[i] + ","
					+ sccdSmoothedErrorRates[i] + "," + sccdrSmoothedErrorRates[i] + "," + refSmoothedErrorRates[i]
					+ ", " + refClassifierSmoothedErrorRates[i]);
		}

		String filePath = "C:/Users/Vincent/Desktop/ErrorRates.csv";

		try {
			Files.write(Paths.get(filePath), errorRatesList);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return measures;
	}

	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss, int numberOfDimensions, int horizon,
			double radius) {
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
			double learningRate = 1;
			adapter = new MCAdapter(horizon, radius, learningRate, "adapting");
			summarisationDescription = "Adapting centroids, horizon: " + horizon + ", radius: " + radius
					+ ", learning rate: " + learningRate;
			break;
		case RADIUSCENTROIDS:
			learningRate = 1;
			adapter = new MCAdapter(horizon, radius, learningRate, "radius");
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