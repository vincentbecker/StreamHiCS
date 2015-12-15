package fullsystem;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

import changechecker.ChangeChecker;
import changechecker.TimeCountChecker;
import contrast.MicroclusterContrast;
import environment.CSVReader;
import clustree.ClusTree;
import moa.streams.ConceptDriftStream;
import streams.GaussianStream;
import subspace.Subspace;
import subspace.SubspaceSet;
import subspacebuilder.AprioriBuilder;
import subspacebuilder.SubspaceBuilder;
import weka.core.Instance;

public class DriftTest {

	private ConceptDriftStream conceptDriftStream;
	private int numberSamples = 0;
	private final int numInstances = 20000;
	private final int numberOfDimensions = 5;
	private double alpha;
	private double epsilon;
	private double threshold;
	private int cutoff;
	private double pruningDifference;
	private CorrelatedSubspacesChangeDetector cscd;
	private FullSpaceChangeDetector refDetector;
	private static CSVReader csvReader;
	private static final String path = "Tests/CovarianceMatrices/";
	//private WithDBSCAN denStream;
	
	//private StreamHiCS streamHiCS;

	@Before
	public void setUp() throws Exception {
		
		csvReader = new CSVReader();
		numberSamples = 0;
		
		alpha = 0.1;
		epsilon = 0.1;
		threshold = 0.5;
		cutoff = 8;
		pruningDifference = 0.15;

		cscd = new CorrelatedSubspacesChangeDetector(numberOfDimensions);
		cscd.alphaOption.setValue(alpha);
		cscd.epsilonOption.setValue(epsilon);
		cscd.thresholdOption.setValue(threshold);
		cscd.cutoffOption.setValue(cutoff);
		cscd.pruningDifferenceOption.setValue(pruningDifference);
		cscd.prepareForUse();
		
		refDetector = new FullSpaceChangeDetector();
		refDetector.prepareForUse();
		
		/*
		ClusTree mcs = new ClusTree();
		mcs.resetLearningImpl();
		Contrast contrastEvaluator = new MicroclusterContrast(50, alpha, mcs);
		ChangeChecker changeChecker = new TimeCountChecker(1000);
		SubspaceBuilder subspaceBuilder = new AprioriBuilder(5, threshold, cutoff, pruningDifference,
				contrastEvaluator);
		//SubspaceBuilder subspaceBuilder = new FastBuilder(5, threshold, pruningDifference, contrastEvaluator);
		this.streamHiCS = new StreamHiCS(epsilon, threshold, pruningDifference, contrastEvaluator, subspaceBuilder, changeChecker,
				null);
		changeChecker.setCallback(streamHiCS);
		*/
		/*
		denStream = new WithDBSCAN();
		denStream.speedOption.setValue(100);
		denStream.epsilonOption.setValue(0.2);
		denStream.betaOption.setValue(0.01);
		denStream.lambdaOption.setValue(0.05);
		denStream.prepareForUse();
		*/
	}
	/*
	@Test
	public void test1() {
		
		GaussianStream s1 = new GaussianStream(null, csvReader.read(path + "Test1.csv"));
		
		double[] mean2 = {5, 5, 5, 5, 5};
		GaussianStream s2 = new GaussianStream(mean2, csvReader.read(path + "Test1.csv"));
		
		conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(s1);
		conceptDriftStream.driftstreamOption.setCurrentObject(s2);
		conceptDriftStream.positionOption.setValue(10000);
		conceptDriftStream.widthOption.setValue(1000);
		conceptDriftStream.prepareForUse();
		
		carryOutTest();
	}
	*/
	@Test
	public void test2() {
		
		GaussianStream s1 = new GaussianStream(null, csvReader.read(path + "Test3.csv"));
		
		double[] mean2 = {5, 5, 5, 5, 5};
		//double[] mean2 = null;
		GaussianStream s2 = new GaussianStream(mean2, csvReader.read(path + "Test2.csv"));
		
		conceptDriftStream = new ConceptDriftStream();
		conceptDriftStream.streamOption.setCurrentObject(s1);
		conceptDriftStream.driftstreamOption.setCurrentObject(s2);
		conceptDriftStream.positionOption.setValue(10000);
		conceptDriftStream.widthOption.setValue(1000);
		conceptDriftStream.prepareForUse();
		
		carryOutTest();
	}

	private void carryOutTest() {
		while (conceptDriftStream.hasMoreInstances() && numberSamples < numInstances) {
			if (numberSamples % 1000 == 0) {
				System.out.println("Time: " + numberSamples);
				System.out.println("Number of microclusters: " + cscd.getNumberOfElements());
				SubspaceSet correlatedSubspaces = cscd.getCurrentlyCorrelatedSubspaces();
				System.out.println("Correlated: " + correlatedSubspaces);
				for (Subspace s : correlatedSubspaces.getSubspaces()) {
					System.out.print(s.getContrast() + ", ");
				}
				System.out.print("\n");
				//System.out.println("Number of microclusters: " + denStream.getMicroClusteringResult().size());
			}
			Instance inst = conceptDriftStream.nextInstance();
			cscd.trainOnInstance(inst);
			refDetector.trainOnInstance(inst);
			//denStream.trainOnInstance(inst);
			/*
			if (cscd.isWarningDetected()) {
				System.out.println("cscd: WARNING at " + numberSamples);
			} else if (cscd.isChangeDetected()) {
				System.out.println("cscd: CHANGE at " + numberSamples);
			}
			
			if (refDetector.isWarningDetected()) {
				System.out.println("refDetector: WARNING at " + numberSamples);
			} else if (refDetector.isChangeDetected()) {
				System.out.println("refDetector: CHANGE at " + numberSamples);
			}
			*/
			numberSamples++;
		}

		fail("Not yet implemented");
	}
}
