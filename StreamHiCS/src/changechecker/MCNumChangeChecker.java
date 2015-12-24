package changechecker;

import fullsystem.Contrast;

/**
 * This class represents a {@link CHangeChecker} which, signals a change when
 * the number of micro-clusters changes significantly (using the Page-Hinkley
 * Test).
 * 
 * @author Vincent
 *
 */
public class MCNumChangeChecker extends ChangeChecker {

	/**
	 * The {@link Contrast} instance.
	 */
	private Contrast contrastEvaluator;

	/**
	 * The magnitude threshold for the Page-Hinkley-Test.
	 */
	private double magnitudeThreshold;

	/**
	 * The detection threshold for the Page-Hinkley-Test.
	 */
	private double detectionThreshold;

	/**
	 * The register of the Page-Hinkley-Test for increasing values.
	 */
	private double increasingMT = Double.MAX_VALUE;

	/**
	 * The register of the Page-Hinkley-Test for decreasing values.
	 */
	private double decreasingMT = Double.MIN_VALUE;

	/**
	 * Creates an instance of this class.
	 * 
	 * @param checkInterval
	 *            The time interval when to check for a change
	 * @param contrastEvaluator
	 *            The {@link Contrast} instance
	 * @param magnitudeThreshold
	 *            The magnitude threshold
	 * @param detectionThreshold
	 *            The detection threshold
	 */
	public MCNumChangeChecker(int checkInterval, Contrast contrastEvaluator, double magnitudeThreshold,
			double detectionThreshold) {
		super(checkInterval);
		this.contrastEvaluator = contrastEvaluator;
		this.magnitudeThreshold = magnitudeThreshold;
		this.detectionThreshold = detectionThreshold;
	}

	@Override
	public boolean checkForChange() {
		double mt = contrastEvaluator.getNumberOfElements() - magnitudeThreshold;

		increasingMT = Math.min(increasingMT, mt);
		decreasingMT = Math.max(decreasingMT, mt);

		double increasingPHT = mt - increasingMT;
		double decreasingPHT = decreasingMT - mt;
		System.out.println("IncreasingPHT: " + increasingPHT + ", DecreasingPHT: " + decreasingPHT);
		if (increasingPHT > detectionThreshold || decreasingPHT > detectionThreshold) {
			reset();
			return true;
		}

		return false;
	}

	/**
	 * Resets the registers of the Page-Hinkley-Test into the initial state. 
	 */
	private void reset() {
		increasingMT = Double.MAX_VALUE;
		decreasingMT = Double.MIN_VALUE;
	}
}
