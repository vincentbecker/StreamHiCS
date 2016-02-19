package changedetection;

/**
 * This interface represents the basic abilities for a concept change detector.
 * 
 * @author Vincent
 *
 */
public interface ChangeDetector {

	/**
	 * Checks whether the change detector is in warning status.
	 * 
	 * @return True, if in warning status, false otherwise.
	 */
	public boolean isWarningDetected();

	/**
	 * Checks whether the change detector has detected a change.
	 * 
	 * @return True, if the change detector has detectoed a change, false
	 *         otherwise.
	 */
	public boolean isChangeDetected();
}
