import React from 'react';
import { createBrowserRouter, RouterProvider, Outlet } from 'react-router-dom';
import ModernHomePage from './pages/ModernHomePage';
import ModernQAPage from './pages/ModernQAPage';
import ModernSearchPage from './pages/ModernSearchPage';
import './ModernApp.css';

function Layout() {
  return (
    <div>
      <Outlet />
    </div>
  );
}

function App() {
  const router = createBrowserRouter([
    {
      path: "/",
      element: <Layout />,
      children: [
        {
          index: true,
          element: <ModernHomePage />
        },
        {
          path: "qa",
          element: <ModernQAPage />
        },
        {
          path: "search",
          element: <ModernSearchPage />
        }
      ]
    }
  ], {
    future: {
      v7_startTransition: true,
      v7_relativeSplatPath: true,
      v7_fetcherPersist: true,
      v7_normalizeFormMethod: true,
      v7_partialHydration: true,
      v7_skipActionErrorRevalidation: true
    }
  });

  return <RouterProvider router={router} />;
}

export default App;