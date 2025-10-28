from campro.freepiston.zerod.cv import (
    ControlVolumeState,
    GasComposition,
    ScavengingState,
    calculate_phase_timing_metrics,
    calculate_scavenging_metrics,
    detect_scavenging_phases,
    enhanced_scavenging_tracking,
)


class TestScavengingState:
    def test_initialization(self):
        """Test ScavengingState initialization."""
        state = ScavengingState()
        assert state.m_fresh_delivered == 0.0
        assert state.m_exhaust_removed == 0.0
        assert state.m_fresh_trapped == 0.0
        assert state.eta_scavenging == 0.0
        assert state.phase_intake_start == 0.0

    def test_update_mass_flows(self):
        """Test mass flow tracking updates."""
        state = ScavengingState()
        composition = GasComposition(fresh_air=0.8, exhaust_gas=0.2)

        state.update_mass_flows(
            mdot_in=0.1, mdot_ex=0.05, dt=0.01, composition=composition,
        )

        assert state.m_fresh_delivered == 0.1 * 0.01
        assert state.m_exhaust_removed == 0.05 * 0.01 * 0.2
        assert state.m_short_circuit == 0.05 * 0.01 * 0.8

    def test_update_trapped_masses(self):
        """Test trapped mass tracking updates."""
        state = ScavengingState()
        composition = GasComposition(fresh_air=0.7, exhaust_gas=0.3)
        cv_state = ControlVolumeState(
            rho=1.0,
            T=1000.0,
            p=1e5,
            u=1000.0,
            h=1500.0,
            composition=composition,
            m=0.1,
            U=100.0,
            H=150.0,
            V=0.1,
            dV_dt=0.0,
        )

        state.update_trapped_masses(cv_state)

        assert state.m_fresh_trapped == 0.1 * 0.7
        assert state.m_exhaust_trapped == 0.1 * 0.3
        assert state.m_total_trapped == 0.1

    def test_calculate_efficiencies(self):
        """Test efficiency calculations."""
        state = ScavengingState()
        state.m_fresh_trapped = 0.07
        state.m_total_trapped = 0.1
        state.m_fresh_delivered = 0.12
        state.m_exhaust_removed = 0.03
        state.m_exhaust_trapped = 0.03
        state.m_short_circuit = 0.01

        state.calculate_efficiencies()

        assert state.eta_scavenging == 0.07 / 0.1
        assert state.eta_trapping == 0.1 / 0.12
        assert state.eta_blowdown == 0.03 / (0.03 + 0.03)
        assert state.eta_short_circuit == 0.01 / 0.12


class TestScavengingPhaseDetection:
    def test_detect_scavenging_phases(self):
        """Test phase detection based on valve areas."""
        A_in_max = 0.01
        A_ex_max = 0.008

        # Both valves closed
        phases = detect_scavenging_phases(0.0, 0.0, A_in_max, A_ex_max)
        assert not phases["intake_open"]
        assert not phases["exhaust_open"]
        assert not phases["overlap"]
        assert phases["both_closed"]

        # Intake only
        phases = detect_scavenging_phases(0.005, 0.0, A_in_max, A_ex_max)
        assert phases["intake_open"]
        assert not phases["exhaust_open"]
        assert not phases["overlap"]
        assert phases["intake_only"]

        # Exhaust only
        phases = detect_scavenging_phases(0.0, 0.004, A_in_max, A_ex_max)
        assert not phases["intake_open"]
        assert phases["exhaust_open"]
        assert not phases["overlap"]
        assert phases["exhaust_only"]

        # Overlap
        phases = detect_scavenging_phases(0.005, 0.004, A_in_max, A_ex_max)
        assert phases["intake_open"]
        assert phases["exhaust_open"]
        assert phases["overlap"]

    def test_phase_detection_threshold(self):
        """Test phase detection with custom threshold."""
        A_in_max = 0.01
        A_ex_max = 0.008

        # Test with higher threshold
        phases = detect_scavenging_phases(
            0.0005, 0.0004, A_in_max, A_ex_max, threshold=0.1,
        )
        assert not phases["intake_open"]
        assert not phases["exhaust_open"]

        # Test with lower threshold
        phases = detect_scavenging_phases(
            0.0005, 0.0004, A_in_max, A_ex_max, threshold=0.01,
        )
        assert phases["intake_open"]
        assert phases["exhaust_open"]


class TestEnhancedScavengingTracking:
    def test_enhanced_scavenging_tracking(self):
        """Test enhanced scavenging tracking with phase detection."""
        # Create test state
        composition = GasComposition(fresh_air=0.8, exhaust_gas=0.2)
        cv_state = ControlVolumeState(
            rho=1.0,
            T=1000.0,
            p=1e5,
            u=1000.0,
            h=1500.0,
            composition=composition,
            m=0.1,
            U=100.0,
            H=150.0,
            V=0.1,
            dV_dt=0.0,
        )

        scavenging_state = ScavengingState()

        # Test with overlap phase
        updated_state = enhanced_scavenging_tracking(
            state=cv_state,
            mdot_in=0.1,
            mdot_ex=0.05,
            A_in=0.005,
            A_ex=0.004,
            A_in_max=0.01,
            A_ex_max=0.008,
            dt=0.01,
            scavenging_state=scavenging_state,
            time=0.1,
        )

        # Check phase timing
        assert updated_state.phase_intake_start == 0.1
        assert updated_state.phase_exhaust_start == 0.1
        assert updated_state.phase_overlap_start == 0.1

        # Check mass flow tracking
        assert updated_state.m_fresh_delivered > 0
        assert updated_state.m_exhaust_removed > 0
        assert updated_state.m_short_circuit > 0

        # Check efficiency calculations
        assert updated_state.eta_scavenging >= 0
        assert updated_state.eta_trapping >= 0
        assert updated_state.eta_blowdown >= 0
        assert updated_state.eta_short_circuit >= 0

    def test_phase_transitions(self):
        """Test phase transition tracking."""
        composition = GasComposition(fresh_air=1.0, exhaust_gas=0.0)
        cv_state = ControlVolumeState(
            rho=1.0,
            T=1000.0,
            p=1e5,
            u=1000.0,
            h=1500.0,
            composition=composition,
            m=0.1,
            U=100.0,
            H=150.0,
            V=0.1,
            dV_dt=0.0,
        )

        scavenging_state = ScavengingState()

        # Start with intake only
        enhanced_scavenging_tracking(
            state=cv_state,
            mdot_in=0.1,
            mdot_ex=0.0,
            A_in=0.005,
            A_ex=0.0,
            A_in_max=0.01,
            A_ex_max=0.008,
            dt=0.01,
            scavenging_state=scavenging_state,
            time=0.1,
        )

        assert scavenging_state.phase_intake_start == 0.1
        assert scavenging_state.phase_exhaust_start == 0.0

        # Transition to overlap
        enhanced_scavenging_tracking(
            state=cv_state,
            mdot_in=0.1,
            mdot_ex=0.05,
            A_in=0.005,
            A_ex=0.004,
            A_in_max=0.01,
            A_ex_max=0.008,
            dt=0.01,
            scavenging_state=scavenging_state,
            time=0.2,
        )

        assert scavenging_state.phase_exhaust_start == 0.2
        assert scavenging_state.phase_overlap_start == 0.2

        # Close intake
        enhanced_scavenging_tracking(
            state=cv_state,
            mdot_in=0.0,
            mdot_ex=0.05,
            A_in=0.0,
            A_ex=0.004,
            A_in_max=0.01,
            A_ex_max=0.008,
            dt=0.01,
            scavenging_state=scavenging_state,
            time=0.3,
        )

        assert scavenging_state.phase_intake_end == 0.3
        assert scavenging_state.phase_overlap_end == 0.3


class TestPhaseTimingMetrics:
    def test_calculate_phase_timing_metrics(self):
        """Test phase timing metrics calculation."""
        scavenging_state = ScavengingState()
        scavenging_state.phase_intake_start = 0.1
        scavenging_state.phase_intake_end = 0.3
        scavenging_state.phase_exhaust_start = 0.2
        scavenging_state.phase_exhaust_end = 0.4
        scavenging_state.phase_overlap_start = 0.2
        scavenging_state.phase_overlap_end = 0.3

        metrics = calculate_phase_timing_metrics(scavenging_state)

        assert abs(metrics["intake_duration"] - 0.2) < 1e-9
        assert abs(metrics["exhaust_duration"] - 0.2) < 1e-9
        assert abs(metrics["overlap_duration"] - 0.1) < 1e-9
        assert abs(metrics["intake_ratio"] - 1.0) < 1e-9
        assert abs(metrics["exhaust_ratio"] - 1.0) < 1e-9
        assert abs(metrics["overlap_ratio"] - 0.5) < 1e-9

    def test_phase_timing_metrics_incomplete_phases(self):
        """Test phase timing metrics with incomplete phases."""
        scavenging_state = ScavengingState()
        scavenging_state.phase_intake_start = 0.1
        # No end time for intake

        metrics = calculate_phase_timing_metrics(scavenging_state)

        assert metrics["intake_duration"] == 0.0
        assert metrics["exhaust_duration"] == 0.0
        assert metrics["overlap_duration"] == 0.0
        assert metrics["intake_ratio"] == 0.0
        assert metrics["exhaust_ratio"] == 0.0
        assert metrics["overlap_ratio"] == 0.0


class TestScavengingMetrics:
    def test_calculate_scavenging_metrics(self):
        """Test basic scavenging metrics calculation."""
        composition = GasComposition(fresh_air=0.8, exhaust_gas=0.2)
        cv_state = ControlVolumeState(
            rho=1.0,
            T=1000.0,
            p=1e5,
            u=1000.0,
            h=1500.0,
            composition=composition,
            m=0.1,
            U=100.0,
            H=150.0,
            V=0.1,
            dV_dt=0.0,
        )

        metrics = calculate_scavenging_metrics(
            state=cv_state,
            mdot_in=0.1,
            mdot_ex=0.05,
            dt=0.01,
        )

        assert abs(metrics["fresh_charge_trapped"] - 0.1 * 0.8) < 1e-9
        assert abs(metrics["total_trapped"] - 0.1) < 1e-9
        assert abs(metrics["scavenging_efficiency"] - 0.8) < 1e-9
        assert abs(metrics["fresh_charge_lost"] - 0.05 * 0.01 * 0.8) < 1e-9

    def test_scavenging_metrics_zero_mass(self):
        """Test scavenging metrics with zero mass."""
        composition = GasComposition(fresh_air=1.0, exhaust_gas=0.0)
        cv_state = ControlVolumeState(
            rho=1.0,
            T=1000.0,
            p=1e5,
            u=1000.0,
            h=1500.0,
            composition=composition,
            m=0.0,
            U=0.0,
            H=0.0,
            V=0.1,
            dV_dt=0.0,
        )

        metrics = calculate_scavenging_metrics(
            state=cv_state,
            mdot_in=0.0,
            mdot_ex=0.0,
            dt=0.01,
        )

        assert metrics["scavenging_efficiency"] == 0.0
        assert metrics["trapping_efficiency"] == 0.0
        assert metrics["short_circuit_loss"] == 0.0


class TestGasComposition:
    def test_gas_composition_normalization(self):
        """Test gas composition normalization."""
        composition = GasComposition(
            fresh_air=0.4, exhaust_gas=0.3, fuel=0.2, burned_gas=0.1,
        )
        composition.normalize()

        total = (
            composition.fresh_air
            + composition.exhaust_gas
            + composition.fuel
            + composition.burned_gas
        )
        assert abs(total - 1.0) < 1e-9

    def test_gas_composition_mixture_properties(self):
        """Test mixture properties calculation."""
        composition = GasComposition(fresh_air=0.8, exhaust_gas=0.2)
        props = composition.get_mixture_properties(T=1000.0, p=1e5)

        assert "MW" in props
        assert "R" in props
        assert "cp" in props
        assert "gamma" in props
        assert props["R"] > 0
        assert props["cp"] > 0
        assert props["gamma"] > 1.0
