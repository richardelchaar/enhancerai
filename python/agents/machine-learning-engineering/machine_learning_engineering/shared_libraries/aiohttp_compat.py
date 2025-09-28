"""
Compatibility module for aiohttp version differences.

This module ensures compatibility between different versions of aiohttp
by providing missing classes that some versions of the Google AI client expect.
"""

import aiohttp

# Fix for aiohttp 3.9+ compatibility with Google AI client
# The Google AI client expects ClientConnectorDNSError which was removed in newer aiohttp versions
if not hasattr(aiohttp, 'ClientConnectorDNSError'):
    # Create ClientConnectorDNSError as an alias to ClientConnectorError
    aiohttp.ClientConnectorDNSError = aiohttp.ClientConnectorError
