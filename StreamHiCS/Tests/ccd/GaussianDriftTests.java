package ccd;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.junit.BeforeClass;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import clustree.ClusTree;
import environment.AccuracyEvaluator;
import environment.CSVReader;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import environment.Stopwatch;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import fullsystem.Callback;
import fullsystem.Contrast;
import fullsystem.CorrelatedSubspacesChangeDetector;
import fullsystem.FullSpaceChangeDetector;
import fullsystem.StreamHiCS;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.streams.ConceptDriftStream;
import streamdatastructures.CentroidsAdapter;
import streamdatastructures.CorrelationSummary;
import streamdatastructures.MicroclusteringAdapter;
import streamdatastructures.SummarisationAdapter;
import streams.GaussianStream;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.ComponentBuilder;
import subspacebuilder.HierarchicalBuilderCutoff;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;
import weka.core.Utils;

public class GaussianDriftTests {
	private ConceptDriftStream conceptDriftStream;
	private StreamHiCS streamHiCS;
	private int numInstances;
	private final int horizon = 1000;
	private final int m = 50;
	private final double alpha = 0.05;
	private double epsilon = 0.15;
	private double aprioriThreshold;
	private double hierarchicalThreshold;
	private double connectedComponentThreshold;
	private int cutoff;
	private double pruningDifference;
	private int numberOfDimensions;
	private Callback callback = new Callback() {
		@Override
		public void onAlarm() {
			// System.out.println("StreamHiCS: onAlarm()");
		}
	};
	private static Stopwatch stopwatch;
	private Contrast contrastEvaluator;
	private static final int numberTestRuns = 1;
	private List<String> results;
	private String summarisationDescription = null;
	private String builderDescription = null;
	private static CSVReader csvReader;
	private static final String path = "Tests/CovarianceMatrices/";
	private double[][] resultSummary;
	private CorrelatedSubspacesChangeDetector cscd;
	private FullSpaceChangeDetector refDetector;

	@BeforeClass
	public static void setUpBeforeClass() {
		stopwatch = new Stopwatch();
		csvReader = new CSVReader();
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
				if (summarisation == StreamSummarisation.ADAPTINGCENTROIDS
						&& buildup == SubspaceBuildup.CONNECTED_COMPONENTS) {
					resultSummary = new double[2][7];
					// int from = 1;
					// int to = 1;
					// int numberTests = from - to + 1;
					for (int test = 3; test <= 3; test++) {
						stopwatch.reset();
						double threshold = 1;

						ArrayList<Double> trueChanges = new ArrayList<Double>();
						results.add("" + test);
						switch (test) {
						case 1:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.5;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.25;
								hierarchicalThreshold = 0.65;
								connectedComponentThreshold = 0.5;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentThreshold;
								break;
							default:
								break;
							}
							numberOfDimensions = 5;
							numInstances = 20000;
							GaussianStream s1 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 0.675);

							GaussianStream s2 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 0.9);

							GaussianStream s3 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.2);

							GaussianStream s4 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.6);

							// GaussianStream s5 = new GaussianStream(null,
							// csvReader.read(path + "Test2.csv"), 0.4);

							ConceptDriftStream conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(4500.0);

							ConceptDriftStream conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(9500.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream.positionOption.setValue(15000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(14500.0);
							/*
							 * conceptDriftStream = new ConceptDriftStream();
							 * conceptDriftStream.streamOption.setCurrentObject(
							 * conceptDriftStream3);
							 * conceptDriftStream.driftstreamOption.
							 * setCurrentObject(s5);
							 * conceptDriftStream.positionOption.setValue(20000)
							 * ; conceptDriftStream.widthOption.setValue(1000);
							 * conceptDriftStream.prepareForUse();
							 * trueChanges.add(19500.0);
							 */
							break;
						case 2:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.25;
								hierarchicalThreshold = 0.25;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.25;
								hierarchicalThreshold = 0.3;
								connectedComponentThreshold = 0.3;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentThreshold;
							default:
								break;
							}
							numberOfDimensions = 5;
							numInstances = 20000;
							s1 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 0.675);

							s2 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 0.9);

							s3 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.2);

							s4 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.6);

							// s5 = new GaussianStream(null, csvReader.read(path
							// + "Test5.csv"), 1.0);

							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(4500.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(9500.0);
							
							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream.positionOption.setValue(15000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(14500.0);
							break;
						case 3:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.4;
								hierarchicalThreshold = 0.55;
								connectedComponentThreshold = 0.5;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.4;
								hierarchicalThreshold = 0.55;
								connectedComponentThreshold = 0.4;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentThreshold;
								break;
							default:
								break;
							}
							numberOfDimensions = 5;
							numInstances = 30000;
							s1 = new GaussianStream(null, csvReader.read(path + "Test1.csv"), 0.675);

							s2 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 0.675);

							s3 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 0.9);

							s4 = new GaussianStream(null, csvReader.read(path + "Test12.csv"), 0.9);

							GaussianStream s5 = new GaussianStream(null, csvReader.read(path + "Test12.csv"), 1.2);

							GaussianStream s6 = new GaussianStream(null, csvReader.read(path + "Test13.csv"), 0.5);

							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							// trueChanges.add(4500.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(9500.0);

							ConceptDriftStream conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream3.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream3.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream3.positionOption.setValue(15000);
							conceptDriftStream3.widthOption.setValue(1000);
							conceptDriftStream3.prepareForUse();
							// trueChanges.add(14500.0);

							ConceptDriftStream conceptDriftStream4 = new ConceptDriftStream();
							conceptDriftStream4.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream4.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream4.positionOption.setValue(20000);
							conceptDriftStream4.widthOption.setValue(1000);
							conceptDriftStream4.prepareForUse();
							trueChanges.add(19500.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream4);
							conceptDriftStream.driftstreamOption.setCurrentObject(s6);
							conceptDriftStream.positionOption.setValue(250000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(24500.0);
							break;
						case 4:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.5;
								hierarchicalThreshold = 0.5;
								connectedComponentThreshold = 0.5;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.45;
								connectedComponentThreshold = 0.5;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentThreshold;
								break;
							default:
								break;
							}
							numberOfDimensions = 10;
							numInstances = 25000;
							s1 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 0.675);

							s2 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 0.9);

							s3 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.2);

							s4 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.6);

							s5 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 2.0);

							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(4500.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(9500.0);

							conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream3.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream3.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream3.positionOption.setValue(15000);
							conceptDriftStream3.widthOption.setValue(1000);
							conceptDriftStream3.prepareForUse();
							trueChanges.add(14500.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream.positionOption.setValue(20000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(19500.0);
							break;
						case 5:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.5;
								hierarchicalThreshold = 0.5;
								connectedComponentThreshold = 0.5;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.5;
								connectedComponentThreshold = 0.5;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentThreshold;
								break;
							default:
								break;
							}
							numberOfDimensions = 10;
							numInstances = 40000;
							s1 = new GaussianStream(null, csvReader.read(path + "Test24.csv"), 0.675);

							s2 = new GaussianStream(null, csvReader.read(path + "Test24.csv"), 0.9);

							s3 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 0.9);

							s4 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.2);

							s5 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.6);

							s6 = new GaussianStream(null, csvReader.read(path + "Test26.csv"), 1.6);

							GaussianStream s7 = new GaussianStream(null, csvReader.read(path + "Test26.csv"), 2);

							GaussianStream s8 = new GaussianStream(null, csvReader.read(path + "Test28.csv"), 2);

							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(4500.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							// trueChanges.add(9500.0);

							conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream3.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream3.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream3.positionOption.setValue(15000);
							conceptDriftStream3.widthOption.setValue(1000);
							conceptDriftStream3.prepareForUse();
							trueChanges.add(14500.0);

							conceptDriftStream4 = new ConceptDriftStream();
							conceptDriftStream4.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream4.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream4.positionOption.setValue(20000);
							conceptDriftStream4.widthOption.setValue(1000);
							conceptDriftStream4.prepareForUse();
							trueChanges.add(19500.0);

							ConceptDriftStream conceptDriftStream5 = new ConceptDriftStream();
							conceptDriftStream5.streamOption.setCurrentObject(conceptDriftStream4);
							conceptDriftStream5.driftstreamOption.setCurrentObject(s6);
							conceptDriftStream5.positionOption.setValue(25000);
							conceptDriftStream5.widthOption.setValue(1000);
							conceptDriftStream5.prepareForUse();
							// trueChanges.add(24500.0);

							ConceptDriftStream conceptDriftStream6 = new ConceptDriftStream();
							conceptDriftStream6.streamOption.setCurrentObject(conceptDriftStream4);
							conceptDriftStream6.driftstreamOption.setCurrentObject(s7);
							conceptDriftStream6.positionOption.setValue(30000);
							conceptDriftStream6.widthOption.setValue(1000);
							conceptDriftStream6.prepareForUse();
							trueChanges.add(29500.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream6);
							conceptDriftStream.driftstreamOption.setCurrentObject(s8);
							conceptDriftStream.positionOption.setValue(35000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							// trueChanges.add(34500.0);
							break;
						case 6:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.5;
								hierarchicalThreshold = 0.5;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.5;
								connectedComponentThreshold = 0.45;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentThreshold;
								break;
							default:
								break;
							}
							numberOfDimensions = 50;
							numInstances = 25000;
							epsilon = 0.2;
							int[] blockBeginnings = {0, 10};
							int[] blockSizes = {10, 10};
							double[][] covarianceMatrix1 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, blockBeginnings, blockSizes, 0.9);
							s1 = new GaussianStream(null, covarianceMatrix1, 0.5);

							s2 = new GaussianStream(null, covarianceMatrix1, 2.0);
							
							s3 = new GaussianStream(null, covarianceMatrix1, 4.0);

							s4 = new GaussianStream(null, covarianceMatrix1, 6.0);

							s5 = new GaussianStream(null, covarianceMatrix1, 8.0);

							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(4500.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();
							trueChanges.add(9500.0);

							conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream3.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream3.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream3.positionOption.setValue(15000);
							conceptDriftStream3.widthOption.setValue(1000);
							conceptDriftStream3.prepareForUse();
							trueChanges.add(14500.0);

							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream.positionOption.setValue(20000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(19500.0);
							break;
						case 7:
							switch (summarisation) {
							case CLUSTREE_DEPTHFIRST:
								aprioriThreshold = 0.5;
								hierarchicalThreshold = 0.5;
								connectedComponentThreshold = 0.45;
								break;
							case ADAPTINGCENTROIDS:
								aprioriThreshold = 0.3;
								hierarchicalThreshold = 0.45;
								connectedComponentThreshold = 0.45;
								break;
							default:
								break;
							}
							switch (buildup) {
							case APRIORI:
								threshold = aprioriThreshold;
								break;
							case HIERARCHICAL:
								threshold = hierarchicalThreshold;
								break;
							case CONNECTED_COMPONENTS:
								threshold = connectedComponentThreshold;
								break;
							default:
								break;
							}
							numberOfDimensions = 50;
							numInstances = 30000;
							epsilon = 0.15;
							int[] blockBeginnings1 = {0, 10};
							int[] blockSizes1 = {10, 10};
							covarianceMatrix1 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, blockBeginnings1, blockSizes1, 0.9);
							s1 = new GaussianStream(null, covarianceMatrix1, 0.5);

							s2 = new GaussianStream(null, covarianceMatrix1, 4.0);
							
							int[] blockBeginnings3 = {20, 30};
							int[] blockSizes3 = {10, 10};
							double[][] covarianceMatrix3 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions, blockBeginnings3, blockSizes3, 0.9);
							s3 = new GaussianStream(null, covarianceMatrix3, 4.0);

							s4 = new GaussianStream(null, covarianceMatrix3, 6.0);

							s5 = new GaussianStream(null, covarianceMatrix1, 6.0);

							s6 = new GaussianStream(null, covarianceMatrix1, 8.0);
							
							conceptDriftStream1 = new ConceptDriftStream();
							conceptDriftStream1.streamOption.setCurrentObject(s1);
							conceptDriftStream1.driftstreamOption.setCurrentObject(s2);
							conceptDriftStream1.positionOption.setValue(5000);
							conceptDriftStream1.widthOption.setValue(1000);
							conceptDriftStream1.prepareForUse();
							trueChanges.add(4500.0);

							conceptDriftStream2 = new ConceptDriftStream();
							conceptDriftStream2.streamOption.setCurrentObject(conceptDriftStream1);
							conceptDriftStream2.driftstreamOption.setCurrentObject(s3);
							conceptDriftStream2.positionOption.setValue(10000);
							conceptDriftStream2.widthOption.setValue(1000);
							conceptDriftStream2.prepareForUse();

							conceptDriftStream3 = new ConceptDriftStream();
							conceptDriftStream3.streamOption.setCurrentObject(conceptDriftStream2);
							conceptDriftStream3.driftstreamOption.setCurrentObject(s4);
							conceptDriftStream3.positionOption.setValue(15000);
							conceptDriftStream3.widthOption.setValue(1000);
							conceptDriftStream3.prepareForUse();
							trueChanges.add(14500.0);
							
							conceptDriftStream4 = new ConceptDriftStream();
							conceptDriftStream4.streamOption.setCurrentObject(conceptDriftStream3);
							conceptDriftStream4.driftstreamOption.setCurrentObject(s5);
							conceptDriftStream4.positionOption.setValue(20000);
							conceptDriftStream4.widthOption.setValue(1000);
							conceptDriftStream4.prepareForUse();
							
							conceptDriftStream = new ConceptDriftStream();
							conceptDriftStream.streamOption.setCurrentObject(conceptDriftStream4);
							conceptDriftStream.driftstreamOption.setCurrentObject(s6);
							conceptDriftStream.positionOption.setValue(25000);
							conceptDriftStream.widthOption.setValue(1000);
							conceptDriftStream.prepareForUse();
							trueChanges.add(24500.0);
							break;
						}

						// Creating the StreamHiCS system
						adapter = createSummarisationAdapter(summarisation);
						contrastEvaluator = new Contrast(m, alpha, adapter);
						CorrelationSummary correlationSummary = new CorrelationSummary(numberOfDimensions, horizon);
						subspaceBuilder = createSubspaceBuilder(buildup, correlationSummary);
						ChangeChecker changeChecker = new TimeCountChecker(1000);
						streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator,
								subspaceBuilder, changeChecker, callback, correlationSummary, stopwatch);
						changeChecker.setCallback(streamHiCS);

						cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions, streamHiCS);
						cscd.prepareForUse();
						streamHiCS.setCallback(cscd);

						refDetector = new FullSpaceChangeDetector();
						AbstractClassifier baseLearner = new HoeffdingTree();
						baseLearner.prepareForUse();
						refDetector.baseLearnerOption.setCurrentObject(baseLearner);
						refDetector.prepareForUse();

						double[] cscdSums = new double[8];
						double[] refSums = new double[8];

						stopwatch.reset();
						for (int i = 0; i < numberTestRuns; i++) {
							System.out.println("Run: " + (i + 1));
							double[][] performanceMeasures = testRun(trueChanges, 1000);
							for (int j = 0; j < 5; j++) {
								cscdSums[j] += performanceMeasures[0][j];
								refSums[j] += performanceMeasures[1][j];
							}
						}

						// Calculate results
						cscdSums[5] = stopwatch.getTime("Evaluation");
						cscdSums[6] = stopwatch.getTime("Adding");
						cscdSums[7] = stopwatch.getTime("Total_CSCD");
						refSums[7] = stopwatch.getTime("Total_REF");

						for (int j = 0; j < 8; j++) {
							cscdSums[j] /= numberTestRuns;
							refSums[j] /= numberTestRuns;
						}

						System.out.println(
								"Test, MTFA, MTD, MDR, MTR, Accuracy, Evaluation time, Adding time, Total time");
						String cscdMeasures = "CSCD," + test + "," + cscdSums[0] + ", " + cscdSums[1] + ", "
								+ cscdSums[2] + ", " + cscdSums[3] + ", " + cscdSums[4] + ", " + cscdSums[5] + ", "
								+ cscdSums[6] + ", " + cscdSums[7];
						String refMeasures = "REF," + test + "," + refSums[0] + ", " + refSums[1] + ", " + refSums[2]
								+ ", " + refSums[3] + ", " + refSums[4] + ", " + refSums[7];
						System.out.println(cscdMeasures);
						System.out.println(refMeasures);
						results.add(cscdMeasures);
						results.add(refMeasures);

						for (int i = 0; i < 7; i++) {
							resultSummary[0][i] += cscdSums[i];
							resultSummary[1][i] += refSums[i];
						}

					}
				}
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/Gaussian/Tests.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private SummarisationAdapter createSummarisationAdapter(StreamSummarisation ss) {
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
			double radius = 4 * Math.sqrt(numberOfDimensions) - 1;
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

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb, CorrelationSummary correlationSummary) {
		cutoff = 8;
		pruningDifference = 0.15;
		boolean addDescription = false;
		if (builderDescription == null) {
			addDescription = true;
		}
		SubspaceBuilder builder = null;
		switch (sb) {
		case APRIORI:
			builder = new AprioriBuilder(numberOfDimensions, aprioriThreshold, cutoff, contrastEvaluator,
					correlationSummary);
			builderDescription = "Apriori, threshold: " + aprioriThreshold + ", cutoff: " + cutoff;
			break;
		case HIERARCHICAL:
			builder = new HierarchicalBuilderCutoff(numberOfDimensions, hierarchicalThreshold, cutoff,
					contrastEvaluator, correlationSummary, true);
			builderDescription = "Hierarchical, threshold: " + hierarchicalThreshold + ", cutoff: " + cutoff;
			break;
		case CONNECTED_COMPONENTS:
			pruningDifference = 0.15;
			builder = new ComponentBuilder(numberOfDimensions, connectedComponentThreshold, contrastEvaluator,
					correlationSummary);
			builderDescription = "Connnected Components, threshold: " + connectedComponentThreshold;
			break;
		default:
			builder = null;
		}
		if (addDescription) {
			results.add(builderDescription);
		}
		return builder;
	}

	private double[][] testRun(ArrayList<Double> trueChanges, int changeLength) {
		conceptDriftStream.restart();
		cscd.resetLearning();
		refDetector.resetLearning();

		int numberSamples = 0;

		ArrayList<Double> cscdDetectedChanges = new ArrayList<Double>();
		ArrayList<Double> refDetectedChanges = new ArrayList<Double>();
		AccuracyEvaluator cscdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator refAccuracy = new AccuracyEvaluator();
		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = conceptDriftStream.nextInstance();

			if(numberSamples % 1000 == 0){
				//System.out.println(cscd.getNumberOfElements());
			}
			
			// For accuracy
			int trueClass = (int) inst.classValue();
			cscdAccuracy.addClassLabel(trueClass);
			cscdAccuracy.addPrediction(cscd.getClassPrediction(inst));
			refAccuracy.addClassLabel(trueClass);
			refAccuracy.addPrediction(Utils.maxIndex(refDetector.getVotesForInstance(inst)));

			stopwatch.start("Total_CSCD");
			cscd.trainOnInstance(inst);
			stopwatch.stop("Total_CSCD");
			stopwatch.start("Total_REF");
			refDetector.trainOnInstance(inst);
			stopwatch.stop("Total_REF");

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
				// System.out.println("refDetector: CHANGE at " +
				// numberSamples);
			}

			numberSamples++;
		}

		double[] cscdPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, cscdDetectedChanges,
				changeLength, numInstances);
		double[] refPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, refDetectedChanges, changeLength,
				numInstances);
		double[][] performanceMeasures = new double[2][5];
		for (int i = 0; i < 4; i++) {
			performanceMeasures[0][i] = cscdPerformanceMeasures[i];
			performanceMeasures[1][i] = refPerformanceMeasures[i];
		}
		performanceMeasures[0][4] = cscdAccuracy.calculateOverallErrorRate();
		performanceMeasures[1][4] = refAccuracy.calculateOverallErrorRate();
		
		String cscdP = "CSCD, ";
		String refP = "REF, ";
		for (int i = 0; i < 5; i++) {
			cscdP += performanceMeasures[0][i] + ", ";
			refP += performanceMeasures[1][i] + ", ";
		}
		//System.out.println(cscdP);
		//System.out.println(refP);
		results.add(cscdP);
		results.add(refP);
		
		List<String> errorRatesList = new ArrayList<String>();
		double[] cscSmoothedErrorRates = cscdAccuracy.calculateSmoothedErrorRates(1000);
		double[] refSmoothedErrorRates = refAccuracy.calculateSmoothedErrorRates(1000);
		for (int i = 0; i < cscdAccuracy.size(); i++) {
			errorRatesList.add(i + "," + cscSmoothedErrorRates[i] + "," + refSmoothedErrorRates[i]);
		}

		String filePath = "C:/Users/Vincent/Desktop/ErrorRates_Gaussian.csv";

		try {
			Files.write(Paths.get(filePath), errorRatesList);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return performanceMeasures;
	}
}
