package changedetection;

import java.util.LinkedList;
import java.util.List;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.meta.WEKAClassifier;
import moa.core.Measurement;
import moa.options.ClassOption;
import weka.core.Instance;
import weka.core.Utils;

public class DriftDetectionMethodClassifier extends AbstractClassifier {

	private static final long serialVersionUID = 1L;

	@Override
	public String getPurposeString() {
		return "Classifier that replaces the current classifier with a new one when a change is detected in accuracy.";
	}

	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l', "Classifier to train.", Classifier.class,
			"bayes.NaiveBayes");

	protected Classifier classifier;

	protected Classifier newclassifier;

	protected DDM driftDetectionMethod;

	protected boolean newClassifierReset;
	// protected int numberInstances = 0;

	protected int ddmLevel;

	/*
	 * public boolean isWarningDetected() { return (this.ddmLevel ==
	 * DriftDetectionMethod.DDM_WARNING_LEVEL); }
	 * 
	 * public boolean isChangeDetected() { return (this.ddmLevel ==
	 * DriftDetectionMethod.DDM_OUTCONTROL_LEVEL); }
	 */

	public static final int DDM_INCONTROL_LEVEL = 0;

	public static final int DDM_WARNING_LEVEL = 1;

	public static final int DDM_OUTCONTROL_LEVEL = 2;

	@Override
	public void resetLearningImpl() {
		this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
		this.newclassifier = this.classifier.copy();
		this.classifier.resetLearning();
		this.newclassifier.resetLearning();
		this.driftDetectionMethod = new DDM();
		this.driftDetectionMethod.prepareForUse();
		this.newClassifierReset = false;
	}

	protected int changeDetected = 0;

	protected int warningDetected = 0;

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// this.numberInstances++;
		int trueClass = (int) inst.classValue();
		boolean prediction;
		if (Utils.maxIndex(this.classifier.getVotesForInstance(inst)) == trueClass) {
			prediction = true;
		} else {
			prediction = false;
		}
		// this.ddmLevel = this.driftDetectionMethod.computeNextVal(prediction);
		this.driftDetectionMethod.input(prediction ? 0.0 : 1.0);
		this.ddmLevel = DDM_INCONTROL_LEVEL;
		if (this.driftDetectionMethod.getChange()) {
			this.ddmLevel = DDM_OUTCONTROL_LEVEL;
		}
		if (this.driftDetectionMethod.getWarningZone()) {
			this.ddmLevel = DDM_WARNING_LEVEL;
		}
		switch (this.ddmLevel) {
		case DDM_WARNING_LEVEL:
			// System.out.println("1 0 W");
			// System.out.println("DDM_WARNING_LEVEL");
			if (newClassifierReset == true) {
				this.warningDetected++;
				this.newclassifier.resetLearning();
				newClassifierReset = false;
			}
			this.newclassifier.trainOnInstance(inst);
			break;

		case DDM_OUTCONTROL_LEVEL:
			// System.out.println("0 1 O");
			// System.out.println("DDM_OUTCONTROL_LEVEL");
			this.changeDetected++;
			this.classifier = null;
			this.classifier = this.newclassifier;
			if (this.classifier instanceof WEKAClassifier) {
				((WEKAClassifier) this.classifier).buildClassifier();
			}
			this.newclassifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
			this.newclassifier.resetLearning();
			break;

		case DDM_INCONTROL_LEVEL:
			// System.out.println("0 0 I");
			// System.out.println("DDM_INCONTROL_LEVEL");
			newClassifierReset = true;
			break;
		default:
			// System.out.println("ERROR!");

		}

		this.classifier.trainOnInstance(inst);
	}

	public double[] getVotesForInstance(Instance inst) {
		return this.classifier.getVotesForInstance(inst);
	}

	@Override
	public boolean isRandomizable() {
		return true;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		((AbstractClassifier) this.classifier).getModelDescription(out, indent);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		List<Measurement> measurementList = new LinkedList<Measurement>();
		measurementList.add(new Measurement("Change detected", this.changeDetected));
		measurementList.add(new Measurement("Warning detected", this.warningDetected));
		Measurement[] modelMeasurements = ((AbstractClassifier) this.classifier).getModelMeasurements();
		if (modelMeasurements != null) {
			for (Measurement measurement : modelMeasurements) {
				measurementList.add(measurement);
			}
		}
		this.changeDetected = 0;
		this.warningDetected = 0;
		return measurementList.toArray(new Measurement[measurementList.size()]);
	}
}
