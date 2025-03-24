import unittest
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from typing import List, Dict, Any, Optional

class TestAnthropicComputerToolHeaders(unittest.TestCase):
    def setUp(self):
        self.anthropic_config = AnthropicConfig()
        
    def test_no_tools(self):
        """Test that None is returned when no tools are provided."""
        result = self.anthropic_config.is_computer_tool_used(tools=None)
        self.assertIsNone(result)
        
        result = self.anthropic_config.is_computer_tool_used(tools=[])
        self.assertIsNone(result)
        
    def test_2024_10_22_tools(self):
        """Test that 'computer-use-2024-10-22' is returned for 2024-10-22 tools."""
        tools = [
            {"type": "computer_20241022", "name": "computer"},
            {"type": "function", "name": "some_function"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2024-10-22")
        
        tools = [
            {"type": "bash_20241022", "name": "bash"},
            {"type": "function", "name": "some_function"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2024-10-22")
        
        tools = [
            {"type": "text_editor_20241022", "name": "editor"},
            {"type": "function", "name": "some_function"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2024-10-22")
        
    def test_2025_01_24_tools(self):
        """Test that 'computer-use-2025-01-24' is returned for 2025-01-24 tools."""
        tools = [
            {"type": "computer_20250124", "name": "computer"},
            {"type": "function", "name": "some_function"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2025-01-24")
        
        tools = [
            {"type": "bash_20250124", "name": "bash"},
            {"type": "function", "name": "some_function"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2025-01-24")
        
        tools = [
            {"type": "text_editor_20250124", "name": "editor"},
            {"type": "function", "name": "some_function"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2025-01-24")
        
    def test_generic_computer_tools(self):
        """Test that 'computer-use-2024-10-22' is returned for generic computer tools."""
        tools = [
            {"type": "computer_other", "name": "computer"},
            {"type": "function", "name": "some_function"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2024-10-22")
        
        tools = [
            {"type": "bash_other", "name": "bash"},
            {"type": "function", "name": "some_function"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2024-10-22")
        
    def test_mixed_tools(self):
        """Test that the newer version is returned when both tool types are present."""
        tools = [
            {"type": "computer_20241022", "name": "computer1"},
            {"type": "computer_20250124", "name": "computer2"}
        ]
        result = self.anthropic_config.is_computer_tool_used(tools=tools)
        self.assertEqual(result, "computer-use-2025-01-24")
        
    def test_headers_generation(self):
        """Test that the headers are correctly generated."""
        # Test 2024-10-22 header
        headers = self.anthropic_config.get_anthropic_headers(
            api_key="test_key",
            computer_tool_used="computer-use-2024-10-22"
        )
        self.assertIn("anthropic-beta", headers)
        self.assertEqual(headers["anthropic-beta"], "computer-use-2024-10-22")
        
        # Test 2025-01-24 header
        headers = self.anthropic_config.get_anthropic_headers(
            api_key="test_key",
            computer_tool_used="computer-use-2025-01-24"
        )
        self.assertIn("anthropic-beta", headers)
        self.assertEqual(headers["anthropic-beta"], "computer-use-2025-01-24")
        
        # Test multiple beta features
        headers = self.anthropic_config.get_anthropic_headers(
            api_key="test_key",
            computer_tool_used="computer-use-2025-01-24",
            prompt_caching_set=True,
            pdf_used=True
        )
        self.assertIn("anthropic-beta", headers)
        self.assertIn("computer-use-2025-01-24", headers["anthropic-beta"])
        self.assertIn("prompt-caching-2024-07-31", headers["anthropic-beta"])
        self.assertIn("pdfs-2024-09-25", headers["anthropic-beta"])

if __name__ == "__main__":
    unittest.main() 