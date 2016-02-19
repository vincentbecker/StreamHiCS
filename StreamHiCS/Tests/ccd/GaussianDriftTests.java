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
import changedetection.FullSpaceChangeDetector;
import changedetection.SubspaceChangeDetectors;
import changedetection.SubspaceClassifiersChangeDetector;
import clustree.ClusTree;
import environment.AccuracyEvaluator;
import environment.CSVReader;
import environment.CovarianceMatrixGenerator;
import environment.Evaluator;
import environment.Stopwatch;
import environment.Parameters.StreamSummarisation;
import environment.Parameters.SubspaceBuildup;
import fullsystem.Contrast;
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
	private static Stopwatch stopwatch;
	private static final int numberTestRuns = 10;
	private List<String> results;
	private String summarisationDescription = null;
	private String builderDescription = null;
	private static CSVReader csvReader;
	private static final String path = "Tests/CovarianceMatrices/";
	private SubspaceChangeDetectors scd;
	private SubspaceClassifiersChangeDetector sccd;
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
		for (StreamSummarisation summarisation : StreamSummarisation.values()) {
			summarisationDescription = null;
			for (SubspaceBuildup buildup : SubspaceBuildup.values()) {
				builderDescription = null;
				if (summarisation == StreamSummarisation.ADAPTINGCENTROIDS
						&& buildup == SubspaceBuildup.CONNECTED_COMPONENTS) {
					for (int test = 7; test <= 7; test++) {
						stopwatch.reset();
						double threshold = 1;

						ArrayList<Double> trueChanges = new ArrayList<Double>();
						ArrayList<Double> trueChangesVirtualD = new ArrayList<Double>();
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
							numberOfDimensions = 5;
							numInstances = 20000;
							GaussianStream s1 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 0.5);

							GaussianStream s2 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.0);

							GaussianStream s3 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.5);

							GaussianStream s4 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 2.0);

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
							
							trueChangesVirtualD.add(4500.0);
							trueChangesVirtualD.add(9500.0);
							trueChangesVirtualD.add(14500.0);
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
							s1 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 0.5);

							s2 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.0);

							s3 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 1.5);

							s4 = new GaussianStream(null, csvReader.read(path + "Test5.csv"), 2.0);

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
							trueChangesVirtualD.add(4500.0);
							trueChangesVirtualD.add(9500.0);
							trueChangesVirtualD.add(14500.0);
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
							numberOfDimensions = 5;
							numInstances = 30000;
							s1 = new GaussianStream(null, csvReader.read(path + "Test1.csv"), 0.5);

							s2 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 0.5);

							s3 = new GaussianStream(null, csvReader.read(path + "Test2.csv"), 1.0);

							s4 = new GaussianStream(null, csvReader.read(path + "Test12.csv"), 1.0);

							GaussianStream s5 = new GaussianStream(null, csvReader.read(path + "Test12.csv"), 1.5);

							GaussianStream s6 = new GaussianStream(null, csvReader.read(path + "Test13.csv"), 2.0);

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
							
							trueChangesVirtualD.add(4500.0);
							trueChangesVirtualD.add(9500.0);
							trueChangesVirtualD.add(14500.0);
							trueChangesVirtualD.add(19500.0);
							trueChangesVirtualD.add(24500.0);
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
							s1 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 0.5);

							s2 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.5);

							s3 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 2.5);

							s4 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 3.5);

							s5 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 4.5);

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
							trueChangesVirtualD.add(4500.0);
							trueChangesVirtualD.add(9500.0);
							trueChangesVirtualD.add(14500.0);
							trueChangesVirtualD.add(19500.0);
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
							numberOfDimensions = 10;
							numInstances = 40000;
							epsilon = 0.15;
							s1 = new GaussianStream(null, csvReader.read(path + "Test24.csv"), 0.5);

							s2 = new GaussianStream(null, csvReader.read(path + "Test24.csv"), 0.9);

							s3 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 0.9);

							s4 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.2);

							s5 = new GaussianStream(null, csvReader.read(path + "Test25.csv"), 1.5);

							s6 = new GaussianStream(null, csvReader.read(path + "Test26.csv"), 1.5);

							GaussianStream s7 = new GaussianStream(null, csvReader.read(path + "Test26.csv"), 1.8);

							GaussianStream s8 = new GaussianStream(null, csvReader.read(path + "Test28.csv"), 1.8);

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
							
							trueChangesVirtualD.add(4500.0);
							trueChangesVirtualD.add(9500.0);
							trueChangesVirtualD.add(14500.0);
							trueChangesVirtualD.add(19500.0);
							trueChangesVirtualD.add(24500.0);
							trueChangesVirtualD.add(29500.0);
							trueChangesVirtualD.add(34500.0);
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
							int[] blockBeginnings = { 0, 10 };
							int[] blockSizes = { 10, 10 };
							double[][] covarianceMatrix1 = CovarianceMatrixGenerator
									.generateCovarianceMatrix(numberOfDimensions, blockBeginnings, blockSizes, 0.9);
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
							
							trueChangesVirtualD.add(4500.0);
							trueChangesVirtualD.add(9500.0);
							trueChangesVirtualD.add(14500.0);
							trueChangesVirtualD.add(19500.0);
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
							int[] blockBeginnings1 = { 0, 10 };
							int[] blockSizes1 = { 10, 10 };
							covarianceMatrix1 = CovarianceMatrixGenerator.generateCovarianceMatrix(numberOfDimensions,
									blockBeginnings1, blockSizes1, 0.9);
							s1 = new GaussianStream(null, covarianceMatrix1, 0.5);

							s2 = new GaussianStream(null, covarianceMatrix1, 4.0);

							int[] blockBeginnings3 = { 20, 30 };
							int[] blockSizes3 = { 10, 10 };
							double[][] covarianceMatrix3 = CovarianceMatrixGenerator
									.generateCovarianceMatrix(numberOfDimensions, blockBeginnings3, blockSizes3, 0.9);
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
							
							trueChangesVirtualD.add(4500.0);
							trueChangesVirtualD.add(9500.0);
							trueChangesVirtualD.add(14500.0);
							trueChangesVirtualD.add(19500.0);
							trueChangesVirtualD.add(24500.0);
							break;
						}

						// Creating the SCD system
						SummarisationAdapter adapter1 = createSummarisationAdapter(summarisation);
						Contrast contrastEvaluator1 = new Contrast(m, alpha, adapter1);
						CorrelationSummary correlationSummary1 = new CorrelationSummary(numberOfDimensions, horizon);
						SubspaceBuilder subspaceBuilder1 = createSubspaceBuilder(buildup, correlationSummary1,
								contrastEvaluator1);
						ChangeChecker changeChecker1 = new TimeCountChecker(1000);
						StreamHiCS streamHiCS1 = new StreamHiCS(epsilon, threshold, pruningDifference,
								contrastEvaluator1, subspaceBuilder1, changeChecker1, null, correlationSummary1,
								stopwatch);
						changeChecker1.setCallback(streamHiCS1);
						scd = new SubspaceChangeDetectors(numberOfDimensions, streamHiCS1);
						scd.useRestspaceOption.setValue(true);
						scd.prepareForUse();
						streamHiCS1.addCallback(scd);

						// Creating the SCCD system
						/*
						SummarisationAdapter adapter2 = createSummarisationAdapter(summarisation);
						Contrast contrastEvaluator2 = new Contrast(m, alpha, adapter2);
						CorrelationSummary correlationSummary2 = new CorrelationSummary(numberOfDimensions, horizon);
						SubspaceBuilder subspaceBuilder2 = createSubspaceBuilder(buildup, correlationSummary2,
								contrastEvaluator2);
						ChangeChecker changeChecker2 = new TimeCountChecker(1000);
						StreamHiCS streamHiCS2 = new StreamHiCS(epsilon, threshold, pruningDifference,
								contrastEvaluator2, subspaceBuilder2, changeChecker2, null, correlationSummary2,
								stopwatch);
						changeChecker2.setCallback(streamHiCS2);
						*/
						sccd = new SubspaceClassifiersChangeDetector(numberOfDimensions, streamHiCS1);
						sccd.useRestspaceOption.setValue(true);
						sccd.prepareForUse();
						streamHiCS1.addCallback(sccd);

						// Creating the reference detector
						refDetector = new FullSpaceChangeDetector();
						AbstractClassifier baseLearner = new HoeffdingTree();
						baseLearner.prepareForUse();
						refDetector.baseLearnerOption.setCurrentObject(baseLearner);
						refDetector.prepareForUse();

						double[] scdSums = new double[12];
						double[] sccdSums = new double[12];
						double[] refSums = new double[12];

						stopwatch.reset();
						for (int i = 0; i < numberTestRuns; i++) {
							System.out.println("Run: " + (i + 1));
							double[][] performanceMeasures = testRun(trueChanges, trueChangesVirtualD,  1000);
							for (int j = 0; j < 9; j++) {
								scdSums[j] += performanceMeasures[0][j];
								sccdSums[j] += performanceMeasures[1][j];
								refSums[j] += performanceMeasures[2][j];
							}
						}

						// Calculate results
						scdSums[9] = stopwatch.getTime("Evaluation");
						scdSums[10] = stopwatch.getTime("Adding");
						scdSums[11] = stopwatch.getTime("Total_SCD");
						sccdSums[9] = stopwatch.getTime("Evaluation");
						sccdSums[10] = stopwatch.getTime("Adding");
						sccdSums[11] = stopwatch.getTime("Total_SCCD");
						refSums[11] = stopwatch.getTime("Total_REF");

						for (int j = 0; j < 12; j++) {
							scdSums[j] /= numberTestRuns;
							sccdSums[j] /= numberTestRuns;
							refSums[j] /= numberTestRuns;
						}

						System.out.println(
								"Test, MTFA, MTD, MDR, MTR, MTFA_V, MTD_V, MTR_V, Error rate, Evaluation time, Adding time, Total time");
						String scdMeasures = "SCD," + test;
						String sccdMeasures = "SCCD," + test;
						String refMeasures = "REF," + test;
						for (int i = 0; i < scdSums.length; i++) {
							scdMeasures += ", " + scdSums[i];
							sccdMeasures += ", " + sccdSums[i];
							refMeasures += ", " + refSums[i];
						}
						System.out.println(scdMeasures);
						System.out.println(sccdMeasures);
						System.out.println(refMeasures);
						results.add(scdMeasures);
						results.add(sccdMeasures);
						results.add(refMeasures);
					}
				}
			}
		}

		// Write the results
		String filePath = "D:/Informatik/MSc/IV/Masterarbeit Porto/Results/ConceptChangeDetection/Gaussian/Tests_new.txt";

		try {
			Files.write(Paths.get(filePath), results, StandardOpenOption.APPEND);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private double[][] testRun(ArrayList<Double> trueChanges, ArrayList<Double> trueChangesVirtualDrift, int changeLength) {
		conceptDriftStream.restart();
		scd.resetLearning();
		sccd.resetLearning();
		refDetector.resetLearning();

		int numberSamples = 0;

		ArrayList<Double> scdDetectedChanges = new ArrayList<Double>();
		ArrayList<Double> sccdDetectedChanges = new ArrayList<Double>();
		ArrayList<Double> refDetectedChanges = new ArrayList<Double>();
		AccuracyEvaluator scdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator sccdAccuracy = new AccuracyEvaluator();
		AccuracyEvaluator refAccuracy = new AccuracyEvaluator();
		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			Instance inst = conceptDriftStream.nextInstance();

			if (numberSamples % 1000 == 0) {
				// System.out.println(cscd.getNumberOfElements());
			}

			// For accuracy
			int trueClass = (int) inst.classValue();
			scdAccuracy.addClassLabel(trueClass);
			scdAccuracy.addPrediction(scd.getClassPrediction(inst));
			sccdAccuracy.addClassLabel(trueClass);
			sccdAccuracy.addPrediction(sccd.getClassPrediction(inst));
			refAccuracy.addClassLabel(trueClass);
			refAccuracy.addPrediction(Utils.maxIndex(refDetector.getVotesForInstance(inst)));

			stopwatch.start("Total_SCD");
			scd.trainOnInstance(inst);
			stopwatch.stop("Total_SCD");
			stopwatch.start("Total_SCCD");
			sccd.trainOnInstance(inst);
			stopwatch.stop("Total_SCCD");
			stopwatch.start("Total_REF");
			refDetector.trainOnInstance(inst);
			stopwatch.stop("Total_REF");

			if (scd.isWarningDetected()) {
				// System.out.println("scd: WARNING at " + numberSamples);
			} else if (scd.isChangeDetected()) {
				scdDetectedChanges.add((double) numberSamples);
				//System.out.println("scd: CHANGE at " + numberSamples);
			}

			if (sccd.isWarningDetected()) {
				// System.out.println("sccd: WARNING at " + numberSamples);
			} else if (sccd.isChangeDetected()) {
				sccdDetectedChanges.add((double) numberSamples);
				//System.out.println("sccd: CHANGE at " + numberSamples);
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

		double[] scdPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, scdDetectedChanges, changeLength,
				numInstances);
		double[] scdPerformanceMeasuresVD = Evaluator.evaluateConceptChange(trueChangesVirtualDrift, scdDetectedChanges,
				changeLength, numInstances);
		double[] sccdPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, sccdDetectedChanges,
				changeLength, numInstances);
		double[] sccdPerformanceMeasuresVD = Evaluator.evaluateConceptChange(trueChangesVirtualDrift,
				sccdDetectedChanges, changeLength, numInstances);
		double[] refPerformanceMeasures = Evaluator.evaluateConceptChange(trueChanges, refDetectedChanges, changeLength,
				numInstances);
		double[] refPerformanceMeasuresVD = Evaluator.evaluateConceptChange(trueChangesVirtualDrift, refDetectedChanges,
				changeLength, numInstances);
		double[][] performanceMeasures = new double[3][9];
		for (int i = 0; i < 4; i++) {
			performanceMeasures[0][i] = scdPerformanceMeasures[i];
			performanceMeasures[1][i] = sccdPerformanceMeasures[i];
			performanceMeasures[2][i] = refPerformanceMeasures[i];
		}
		for (int i = 4; i < 8; i++) {
			performanceMeasures[0][i] = scdPerformanceMeasuresVD[i - 4];
			performanceMeasures[1][i] = sccdPerformanceMeasuresVD[i - 4];
			performanceMeasures[2][i] = refPerformanceMeasuresVD[i - 4];
		}
		performanceMeasures[0][8] = scdAccuracy.calculateOverallErrorRate();
		performanceMeasures[1][8] = sccdAccuracy.calculateOverallErrorRate();
		performanceMeasures[2][8] = refAccuracy.calculateOverallErrorRate();
		String scdP = "SCD";
		String sccdP = "SCCD";
		String refP = "REF";
		for (int i = 0; i < 9; i++) {
			scdP += ", " + performanceMeasures[0][i];
			sccdP += ", " + performanceMeasures[1][i];
			refP += ", " + performanceMeasures[2][i];
		}
		System.out.println(scdP);
		System.out.println(sccdP);
		System.out.println(refP);
		results.add(scdP);
		results.add(sccdP);
		results.add(refP);

		List<String> errorRatesList = new ArrayList<String>();
		double[] scdSmoothedErrorRates = scdAccuracy.calculateSmoothedErrorRates(1000);
		double[] sccdSmoothedErrorRates = sccdAccuracy.calculateSmoothedErrorRates(1000);
		double[] refSmoothedErrorRates = refAccuracy.calculateSmoothedErrorRates(1000);
		for (int i = 0; i < scdAccuracy.size(); i++) {
			errorRatesList.add(i + "," + scdSmoothedErrorRates[i] + "," + sccdSmoothedErrorRates[i] + ","
					+ refSmoothedErrorRates[i]);
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

	private SubspaceBuilder createSubspaceBuilder(SubspaceBuildup sb, CorrelationSummary correlationSummary,
			Contrast contrastEvaluator) {
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
}
