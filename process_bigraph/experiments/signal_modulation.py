import numpy as np
from process_bigraph.composite import Process, Composite
from process_bigraph.registry import process_registry
# import matplotlib.pyplot as plt


class MediumDistortionProcess(Process):
    config_schema = {
        'input_signal': 'list[float]',
    }

    def __init__(self, config=None):
        super().__init__(config)

    def initial_state(self):
        return {'output_signal': self.config['input_signal']}

    def schema(self):
        return {
            'output_signal': 'list[float]',
        }

    def update(self, state, interval):
        new_wave = apply_modulation(state['output_signal'], distortion, gain=5).tolist()
        return {
            'output_signal': new_wave
        }


process_registry.register('medium_distortion', MediumDistortionProcess)


def apply_modulation(input_wave, modulation_function, **kwargs):
    """
    Apply a modulation effect to an input wave.

    :param input_wave: NumPy array, the input waveform.
    :param modulation_function: function, the modulation effect to apply.
    :param kwargs: additional keyword arguments for the modulation function.
    :return: NumPy array, the modulated waveform.
    """
    return modulation_function(input_wave, **kwargs)


def distortion(input_wave, gain=1):
    """
    Apply a simple distortion effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param gain: float, the gain factor for distortion.
    :return: NumPy array, the distorted waveform.
    """
    return np.clip(input_wave * gain, -1, 1)


def tremolo(input_wave, rate=5, depth=0.5):
    """
    Apply a tremolo effect to the waveform.

    :param input_wave: NumPy array, the input waveform.
    :param rate: float, the rate of the tremolo effect.
    :param depth: float, the depth of the tremolo effect.
    :return: NumPy array, the waveform with tremolo effect.
    """
    t = np.linspace(0, 1, len(input_wave), endpoint=True)
    modulating_wave = (1 - depth) + depth * np.sin(2 * np.pi * rate * t)
    return input_wave * modulating_wave


'''# Example usage
t = np.linspace(0, 1, 500, endpoint=True)
input_wave = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500, endpoint=True))  # Example sine wave

# Apply distortion
modulated_wave = apply_modulation(input_wave, distortion, gain=5)
print(modulated_wave)
# Apply tremolo
modulated_wave = apply_modulation(modulated_wave, tremolo, rate=5, depth=0.7)
print(modulated_wave)'''


'''# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, input_wave, label='Original Wave')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, distorted_wave, label='Distorted Wave')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, tremolo_wave, label='Tremolo Wave')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()'''


def test_medium_distortion():
    instance = {
            'distortion': {
                '_type': 'process',
                'address': 'local:medium_distortion',
                'config': {
                    'input_signal': np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500, endpoint=True)),
                },
                'wires': {  # this should return that which is in the schema
                    'output_signal': ['output_signal_store'],
                }
            },
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'ports': {
                        'inputs': {
                            'output_signal': 'list[float]'
                        },
                    }
                },
                'wires': {
                    'inputs': {
                        'output_signal': ['output_signal_store'],
                    }
                }
            }
        }

    # make the composite
    workflow = Composite({
        'state': instance
    })

    # run
    workflow.run(10)

    # gather results
    results = workflow.gather_results()
    print(results)


test_medium_distortion()
