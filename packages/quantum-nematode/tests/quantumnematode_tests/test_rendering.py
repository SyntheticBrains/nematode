"""Tests for the episode rendering system."""

from unittest.mock import MagicMock, patch

from quantumnematode.rendering import EpisodeRenderer


class TestEpisodeRendererInitialization:
    """Test episode renderer initialization."""

    def test_initialize_enabled_by_default(self):
        """Test that renderer is enabled by default."""
        renderer = EpisodeRenderer()

        assert renderer.enabled is True

    def test_initialize_disabled(self):
        """Test initialization with rendering disabled."""
        renderer = EpisodeRenderer(enabled=False)

        assert renderer.enabled is False


class TestRenderIfNeeded:
    """Test conditional rendering logic."""

    def test_render_all_frames_mode(self):
        """Test rendering when show_last_frame_only is False."""
        renderer = EpisodeRenderer()
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with patch("builtins.print"):
            # Render step 0 of 100
            renderer.render_if_needed(
                mock_env,
                step=0,
                max_steps=100,
                show_last_frame_only=False,
            )

            # Should have rendered
            assert mock_env.render.called

    def test_render_last_frame_only_mode_middle_step(self):
        """Test that middle frames are skipped in last-frame-only mode."""
        renderer = EpisodeRenderer()
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with patch("builtins.print"):
            # Render step 50 of 100
            renderer.render_if_needed(
                mock_env,
                step=50,
                max_steps=100,
                show_last_frame_only=True,
            )

            # Should NOT have rendered
            assert not mock_env.render.called

    def test_render_last_frame_only_mode_last_step(self):
        """Test that last frame is rendered in last-frame-only mode."""
        renderer = EpisodeRenderer()
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with patch("builtins.print"):
            # Render step 99 of 100 (last step, 0-indexed)
            renderer.render_if_needed(
                mock_env,
                step=99,
                max_steps=100,
                show_last_frame_only=True,
            )

            # Should have rendered
            assert mock_env.render.called

    def test_render_disabled(self):
        """Test that disabled renderer does not render."""
        renderer = EpisodeRenderer(enabled=False)
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with patch("builtins.print"):
            renderer.render_if_needed(
                mock_env,
                step=0,
                max_steps=100,
                show_last_frame_only=False,
            )

            # Should NOT have rendered
            assert not mock_env.render.called

    def test_render_with_text(self):
        """Test rendering with additional text."""
        renderer = EpisodeRenderer()
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with patch("builtins.print") as mock_print:
            renderer.render_if_needed(
                mock_env,
                step=0,
                max_steps=100,
                text="Episode 1",
            )

            # Should have printed both grid and text
            assert mock_print.call_count >= 2


class TestRenderFrame:
    """Test single frame rendering."""

    def test_render_frame_basic(self):
        """Test basic frame rendering."""
        renderer = EpisodeRenderer()
        mock_env = MagicMock()
        mock_env.render.return_value = "test grid"

        with patch("builtins.print") as mock_print:
            renderer.render_frame(mock_env)

            # Should have called env.render()
            mock_env.render.assert_called_once()
            # Should have printed the grid
            mock_print.assert_called()

    def test_render_frame_with_text(self):
        """Test rendering frame with text."""
        renderer = EpisodeRenderer()
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with patch("builtins.print") as mock_print:
            renderer.render_frame(mock_env, text="Test Text")

            # Should print both grid and text
            assert mock_print.call_count >= 2

    def test_render_frame_no_clear(self):
        """Test rendering without clearing screen."""
        renderer = EpisodeRenderer()
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with (
            patch("builtins.print"),
            patch.object(
                EpisodeRenderer,
                "clear_screen",
            ) as mock_clear,
        ):
            renderer.render_frame(mock_env, clear_screen=False)

            # Should NOT have cleared screen
            mock_clear.assert_not_called()

    def test_render_frame_with_clear(self):
        """Test rendering with screen clearing."""
        renderer = EpisodeRenderer()
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with (
            patch("builtins.print"),
            patch.object(
                EpisodeRenderer,
                "clear_screen",
            ) as mock_clear,
        ):
            renderer.render_frame(mock_env, clear_screen=True)

            # Should have cleared screen
            mock_clear.assert_called_once()

    def test_render_frame_disabled(self):
        """Test that disabled renderer does not render frames."""
        renderer = EpisodeRenderer(enabled=False)
        mock_env = MagicMock()
        mock_env.render.return_value = "grid"

        with patch("builtins.print") as mock_print:
            renderer.render_frame(mock_env)

            # Should NOT have rendered
            mock_env.render.assert_not_called()
            mock_print.assert_not_called()


class TestClearScreen:
    """Test screen clearing functionality."""

    @patch("sys.platform", "linux")
    @patch("builtins.print")
    def test_clear_screen_unix(self, mock_print):
        """Test screen clearing on Unix-like systems."""
        EpisodeRenderer.clear_screen()

        # Should use ANSI escape sequence
        mock_print.assert_called_once_with("\033[2J\033[H", end="")

    @patch("sys.platform", "darwin")
    @patch("builtins.print")
    def test_clear_screen_macos(self, mock_print):
        """Test screen clearing on macOS."""
        EpisodeRenderer.clear_screen()

        # Should use ANSI escape sequence
        mock_print.assert_called_once()

    @patch("sys.platform", "win32")
    @patch("os.system")
    def test_clear_screen_windows(self, mock_system):
        """Test screen clearing on Windows."""
        EpisodeRenderer.clear_screen()

        # Should use cls command
        mock_system.assert_called_once_with("cls")
